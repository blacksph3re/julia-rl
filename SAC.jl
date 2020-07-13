module rl

using Flux
using Zygote
using PyCall
using DataStructures
using StatsBase
using Printf
using LinearAlgebra
using Distributions
using DistributionsAD # For differentiable logpdf
using Test
using BSON
using CUDA

#CUDA.allowscalar(false)

layer_size = 256

hparams = Dict([
    ("lr", 1e-3),
    ("env", "CartPoleContinuousBulletEnv-v0"),
    ("policy_layer1", layer_size),
    ("policy_layer2", layer_size),
    ("value_layer1", layer_size),
    ("value_layer2", layer_size),
    ("q_layer1", layer_size),
    ("q_layer2", layer_size),
    ("activation", swish),
    ("target_update", 1e-3),
    ("entropy_incentive", 0.2),
    ("l2_reg", 1e-4),
    ("batch_size", 100),
    ("discount_factor", 0.99),
    ("buffer_size", 1000000),
    ("epochs", 1000),
    ("steps_per_epoch", 2000),
    ("train_steps_per_iter", 2)
])


dtype = Float32

pyimport("pybullet_envs")
gym = pyimport("gym")
env = gym.make(hparams["env"])


STATE_SPACE = length(env.observation_space.low)
ACTION_SPACE = length(env.action_space.low)
ACTION_HIGH = env.action_space.high
ACTION_LOW = env.action_space.low
ACTION_HIGH_GPU = gpu(dtype.(ACTION_HIGH))
ACTION_LOW_GPU = gpu(dtype.(ACTION_LOW))
TARGET_UPDATE = gpu(dtype(hparams["target_update"]))
ENTROPY_INCENTIVE = gpu(dtype(hparams["entropy_incentive"]))
L2_REG = hparams["l2_reg"]
BATCH_SIZE = hparams["batch_size"]
GAMMA = gpu(dtype(hparams["discount_factor"]))
EPOCHS = hparams["epochs"]
STEPS_PER_EPOCH = hparams["steps_per_epoch"]
TRAIN_STEPS_PER_ITER = hparams["train_steps_per_iter"]

optim = Flux.ADAM(hparams["lr"])

# Our squashed policy has a support of -1,1
# Project actions to/from that support
function action_to_support(action)
    halfspan = (ACTION_HIGH_GPU .- ACTION_LOW_GPU) ./ 2
    low_end = ACTION_LOW_GPU ./ halfspan
    action ./ halfspan .- low_end .- 1
end

function support_to_action(action)
    halfspan = (ACTION_HIGH_GPU .- ACTION_LOW_GPU) ./ 2
    low_end = ACTION_LOW_GPU ./ halfspan
    (action .+ 1 .+ low_end) .* halfspan
end

#begin
#    local lol1 = Flux.batch([rand(ACTION_SPACE) for _ in 1:100])
#    @test isapprox(lol1, support_to_action(action_to_support(lol1)))
#end


function initialize()
    
    typeswitch(T, x) = x
    typeswitch(T, x::Number) = T(x)
    typeswitch(T, x::AbstractArray) = T.(x)
    
    value = Chain(
        Dense(STATE_SPACE, hparams["value_layer1"], hparams["activation"]),
        #LayerNorm(hparams["value_layer1"]),
        Dense(hparams["value_layer1"], hparams["value_layer2"], hparams["activation"]),
        #LayerNorm(hparams["value_layer2"]),
        Dense(hparams["value_layer2"], 1),
    )
    value = Flux.fmap(x -> typeswitch(dtype, x), value)

    value_target = deepcopy(value)

    critics = map((_) -> Flux.fmap(x -> typeswitch(dtype, x), Chain(
        Dense(STATE_SPACE+ACTION_SPACE, hparams["q_layer1"], hparams["activation"]),
        #LayerNorm(hparams["q_layer1"]),
        Dense(hparams["q_layer1"], hparams["q_layer2"], hparams["activation"]),
        #LayerNorm(hparams["q_layer2"]),
        Dense(hparams["q_layer2"], 1),
    )), 1:2)

    policy = Chain(
        Dense(STATE_SPACE, hparams["policy_layer1"], hparams["activation"]),
        #LayerNorm(hparams["policy_layer1"]),
        Dense(hparams["policy_layer1"], hparams["policy_layer2"], hparams["activation"]),
        #LayerNorm(hparams["policy_layer2"]),
        Dense(hparams["policy_layer2"], ACTION_SPACE*2),
    )
    policy = Flux.fmap(x -> typeswitch(dtype, x), policy)

    
    memory = memory = CircularBuffer{Tuple{AbstractArray{dtype,1}, AbstractArray{dtype, 1}, dtype, AbstractArray{dtype,1}, Bool}}(hparams["buffer_size"])
    
    gpu(value), gpu(value_target), gpu(critics), gpu(policy), memory
end




# Single value mode
function forward_critic(models, state::AbstractArray{T,1}, action::AbstractArray{T,1}) where {T}
    pred = map((model) -> model(vcat(state, action)), models)
    minimum(Flux.stack(pred, 1))
end
# Batch mode
function forward_critic(models, state::AbstractArray{T,2}, action::AbstractArray{T,2}) where {T}
    input = vcat(state, action)
    #pred = map((model) -> dropdims(model(), dims=1), models)
    minimum(Flux.stack([
        models[1](input),
        models[2](input)
    ], 1), dims=1)
end



function forward_policy(model, state::AbstractArray{T, 2}) where {T}
    x = policy(state)
    mu = x[1:ACTION_SPACE,:]
    sigma = exp.(x[ACTION_SPACE+1:end,:])
    return mu, sigma
end
function forward_policy(model, state::AbstractArray{T, 1}) where {T}
    x = policy(state)
    mu = x[1:ACTION_SPACE]
    sigma = exp.(x[ACTION_SPACE+1:end])
    return mu, sigma
end

# The SAC authors squash their action space
# See Appendix C in their paper
function squash(actions)
    tanh.(actions)
end
function policy_sample(mu::AbstractArray{T}, sigma::AbstractArray{T}) where {T}
    mu .+ sigma .* gpu(clamp.(T.(rand(Normal(), size(sigma))), -10, 10))
end
function policy_sample(model, state::AbstractArray)
    policy_sample(forward_policy(model, state)...)
end
function act(model, state)
    support_to_action(squash(policy_sample(model, state)))
end
function logprob(mu::AbstractArray{T,1}, sigma::AbstractArray{T,1}, action_unsquashed::AbstractArray{T,1}) where {T}
    # Add a small epsilon so we don't explode the loss
    T(logpdf(TuringDiagMvNormal(mu, sigma), action_unsquashed)) - sum(log.(1 .- tanh.(action_unsquashed) .^ 2 .+ eps(T) ))
end
#function logprobs(mu::AbstractArray{T,2}, sigma::AbstractArray{T,2}, actions_unsquashed::AbstractArray{T,2}) where {T}
#    slicemap(
#	(x) -> [logprob(x[1:ACTION_SPACE], x[ACTION_SPACE+1:ACTION_SPACE*2], x[ACTION_SPACE*2+1:end])],
#	[mu;sigma;actions_unsquashed],
#	dims=1)
#end

# This version has a performance problem because the results are sent back to CPU
function logprobs(mu::AbstractArray{T,2}, sigma::AbstractArray{T,2}, actions_unsquashed::AbstractArray{T,2}) where {T}
    gpu(map((i) -> logprob(mu[:,i], sigma[:,i], actions_unsquashed[:,i]), 1:BATCH_SIZE))
end

# This version should work well on batches
function logprobs_cuda(mu::AbstractArray{T, 2}, sigma::AbstractArray{T, 2}, x::AbstractArray{T, 2}) where {T}
    # A pretty intimidating function
    # The first half is the normal logpdf part, copied from DistributionsAD for a diagonal multivariate gaussian where the sums are respecting batching
    # The second part is needed because the function is squashed with tanh
    -(size(mu, 1) * log(2*pi) .+ 2 * sum(log.(sigma), dims=1) .+ sum(((x .- mu) ./ sigma) .^ 2, dims=1)) / 2 .- sum(log.(1 .- tanh.(x) .^ 2 .+ eps(T)), dims=1)
end


function update_value_target!(value_target, value)
    for (p, p_target) in zip(Flux.params(value), Flux.params(value_target))
        p_target .= (1-TARGET_UPDATE) .* p_target .+ TARGET_UPDATE .* p
        @assert !any(isnan.(p_target))
    end
end

function update_value!(value, critics, policy, optim, states::AbstractArray{T}) where {T}
    parameters = Flux.params(value)
    outer_loss = 0
    
    mu, sigma = forward_policy(policy, states)
    actions = policy_sample(mu, sigma)
    p = logprobs_cuda(mu, sigma, actions)
    q = forward_critic(critics, states, squash(actions))
    
    target = q .- ENTROPY_INCENTIVE .* p
    
    grads = gradient(parameters) do
        loss = Flux.mse(value(states), target)# + L2_REG * sum(norm, parameters)
        outer_loss += loss
        return loss
    end
    
    Flux.update!(optim, parameters, grads)
    #@assert !any([any(isnan.(layer.W)) for layer in value])
    outer_loss
end

function update_critic!(critic, value_target, optim, states, actions, rewards, next_states, deaths)
    parameters = Flux.params(critic)
    outer_loss = 0
    target = rewards .+ GAMMA .* (1 .- deaths) .* dropdims(value_target(next_states), dims=1)
        
    grads = gradient(parameters) do
        loss = Flux.mse(critic(vcat(states, actions)), target)# + L2_REG * sum(norm, parameters)
        outer_loss += loss
        return loss
    end
    
    Flux.update!(optim, parameters, grads)
    
    #@assert !any([any(isnan.(layer.W)) for layer in critic])
    outer_loss
end

function update_policy!(policy, critics, optim, states::AbstractArray{T}) where {T}
    parameters = Flux.params(policy)
    outer_loss = 0
    batch_size = size(states)[2]
    unitnoise = T.(rand(Normal(), (ACTION_SPACE, batch_size)))
    unitnoise = gpu(clamp.(unitnoise, -10, 10)) # Don't allow crazy outliers
    
    grads = gradient(parameters) do
        mu, sigma = forward_policy(policy, states)
        actions = mu .+ sigma .* unitnoise
        p = ENTROPY_INCENTIVE * logprobs_cuda(mu, sigma, actions)
        q = forward_critic(critics, states, squash(actions))
        loss = sum(p .- q) / batch_size #+ L2_REG * sum(norm, parameters)
        outer_loss += loss
        return loss
    end
        
    Flux.update!(optim, parameters, grads)
    #@assert !any([any(isnan.(layer.W)) for layer in policy])
    outer_loss
end

function batch(memory)
    batch = vcat([sample(memory) for _ in 1:BATCH_SIZE])
    
    # Destructure the batch
    state, action, reward, next_state, death = [getindex.(batch, i) for i in 1:5]
    state = Flux.batch(state)
    next_state = Flux.batch(next_state)
    action = Flux.batch(action)
    death = Int64.(death)
    
    return gpu(state), gpu(action), gpu(reward), gpu(next_state), gpu(death)
end



function collect_experience!(env, memory, steps)
  state = dtype.(env.reset())
  total_reward = 0
  total_deaths = 0
  for _ in 1:steps
    action = dtype.(clamp(rand(MvNormal(ACTION_LOW, ACTION_HIGH)), ACTION_LOW, ACTION_HIGH))
    next_state, reward, death, _ = env.step(action) # Advance the env

    # Convert to dtype
    next_state = dtype.(next_state)
    reward = dtype(reward)
    total_reward += reward

    push!(memory, (state, action, reward, next_state, death))
    
    if death
        state = env.reset()
        total_deaths += 1
    end
  end
  total_reward / (total_deaths + 1)
end



function train!(env, memory, optim, policy, critics, value, value_target)
  for epoch in 1:EPOCHS
    time = @elapsed begin
        total_reward = 0
        total_deaths = 0
        total_v_loss = 0
        total_q1_loss = 0
        total_q2_loss = 0
        total_policy_loss = 0
        total_entropy = 0
        total_iterations = 0
        state = dtype.(env.reset())
        for i in 1:STEPS_PER_EPOCH
            action = cpu(act(policy, gpu(state))) # Act
            next_state, reward, death, _ = env.step(action) # Advance the env

            # Convert to Float32
            next_state = dtype.(next_state)
            reward = dtype(reward)

            push!(memory, (state, action, reward, next_state, death))
            total_reward += reward
            if death
                state = env.reset()
                total_deaths += 1
            else
                state = next_state
            end

            if length(memory) > BATCH_SIZE
                for i in 1:1
                    states, actions, rewards, next_states, deaths = batch(memory)
                    actions = action_to_support(actions)
                    total_v_loss += update_value!(value, critics, policy, optim, states)
                    total_q1_loss += update_critic!(critics[1], value_target, optim, states, actions, rewards, next_states, deaths)
                    total_q2_loss += update_critic!(critics[2], value_target, optim, states, actions, rewards, next_states, deaths)
                    total_policy_loss += update_policy!(policy, critics, optim, states)
                    update_value_target!(value_target, value)
                    @assert(!isnan(total_v_loss))
                    @assert(!isnan(total_q1_loss))
                    @assert(!isnan(total_q2_loss))
                    @assert(!isnan(total_policy_loss))
                    total_iterations += 1
                end
            end
        end
    end

    println(@sprintf("I: %d, r: %f, v: %f, q1: %f, q2: %f, p: %f, t: %f",
            epoch,
            total_reward/(total_deaths+1),
            total_v_loss/total_iterations,
            total_q1_loss/total_iterations,
            total_q2_loss/total_iterations,
            total_policy_loss/total_iterations,
            time))

    s,a,r,sn,d = batch(memory)
    v = mean(value(s))
    q1 = mean(critics[1](vcat(s, a)))
    q2 = mean(critics[2](vcat(s, a)))
    mu, sigma = forward_policy(policy, s)
    println(@sprintf("   v: %f, q1: %f, q2: %f, e: %f", 
            v, 
            q1,
            q2,
            mean(sigma)))

    
    flush(stdout)
  end
end


value, value_target, critics, policy, memory = initialize()
collect_experience!(env, memory, 200)

train!(env, memory, optim, policy, critics, value, value_target)


function test(env, policy)
    state = env.reset()
    total_reward = 0
    for _ in 1:5000
        mu, sigma = forward_policy(policy, state)
        state, reward, death, _ = env.step(mu)
        total_reward += reward
        env.render()
        if death
            break
        end
    end
    total_reward
end

test(env, policy)

@BSON.save "models.bson" Dict(
    :value => value,
    :value_target => value_target,
    :critics => critics,
    :policy => policy,
)

BATCH_SIZE = 10000
s,a,r,sa,d = batch(memory)

mean(r)

end
