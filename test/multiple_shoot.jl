using DiffEqFlux, Lux, Optimization, OptimizationFlux, OrdinaryDiffEq, Test, Random
using DiffEqFlux: group_ranges

rng = Random.default_rng()
## Test group partitioning helper function
@test group_ranges(10, 4) == [1:4, 4:7, 7:10]
@test group_ranges(10, 5) == [1:5, 5:9, 9:10]
@test group_ranges(10, 10) == [1:10]
@test_throws DomainError group_ranges(10, 1)
@test_throws DomainError group_ranges(10, 11)

## Define initial conditions and time steps
datasize = 30
u0 = Float32[2.0, 0.0]
tspan = (0.0f0, 5.0f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

# Get the data
function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end
prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

# Define the Neural Network
nn = Lux.Chain(ActivationFunction(x -> x.^3),
                Lux.Dense(2, 16, tanh),
                Lux.Dense(16, 2))
p_init, st = Lux.setup(rng, nn)
p_init = Lux.ComponentArray(p_init)

neuralode = NeuralODE(nn, tspan, Tsit5(), saveat = tsteps)
prob_node = ODEProblem((u,p,t)->nn(u,p,st)[1], u0, tspan, p_init)

function predict_single_shooting(p)
    return Array(neuralode(u0, p, st)[1])
end

# Define loss function
function loss_function(data, pred)
	return sum(abs2, data - pred)
end

## Evaluate Single Shooting
function loss_single_shooting(p)
    pred = predict_single_shooting(p)
    l = loss_function(ode_data, pred)
    return l, pred
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((p)->loss_single_shooting(p), adtype)
optprob = Optimization.OptimizationProblem(optf, p_init)
res_single_shooting = Optimization.solve(loss_single_shooting, p_init,
                                          ADAM(0.05),
										  maxiters = 300)

loss_ss, _ = loss_single_shooting(res_single_shooting.minimizer)
println("Single shooting loss: $(loss_ss)")

## Test Multiple Shooting
group_size = 3
continuity_term = 200

function loss_multiple_shooting(p)
    return multiple_shoot(p, ode_data, tsteps, prob_node, loss_function, Tsit5(),
                          group_size; continuity_term,
                          abstol=1e-8, reltol=1e-6) # test solver kwargs
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((p)->loss_multiple_shooting(p), adtype)
optprob = Optimization.OptimizationProblem(optf, p_init)
res_ms = Optimization.solve(optprob,
                            ADAM(0.05), maxiters = 300)

# Calculate single shooting loss with parameter from multiple_shoot training
loss_ms, _ = loss_single_shooting(res_ms.minimizer)
println("Multiple shooting loss: $(loss_ms)")
@test loss_ms < 10loss_ss

# Test with custom loss function
group_size = 4
continuity_term = 50

function continuity_loss_abs2(û_end, u_0)
    return sum(abs2, û_end - u_0) # using abs2 instead of default abs
end

function loss_multiple_shooting_abs2(p)
    return multiple_shoot(p, ode_data, tsteps, prob_node,
                          loss_function, continuity_loss_abs2, Tsit5(),
                          group_size; continuity_term)
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((p)->loss_multiple_shooting_abs2(p), adtype)
optprob = Optimization.OptimizationProblem(optf, p_init)
res_ms_abs2 = Optimization.solve(optprob,
                                     ADAM(0.05), maxiters = 300)

loss_ms_abs2, _ = loss_single_shooting(res_ms_abs2.minimizer)
println("Multiple shooting loss with abs2: $(loss_ms_abs2)")
@test loss_ms_abs2 < loss_ss

## Test different SensitivityAlgorithm (default is InterpolatingAdjoint)
function loss_multiple_shooting_fd(p)
    return multiple_shoot(p, ode_data, tsteps, prob_node,
                          loss_function, continuity_loss_abs2, Tsit5(),
                          group_size; continuity_term,
                          sensealg=ForwardDiffSensitivity())
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((p)->loss_multiple_shooting_fd(p), adtype)
optprob = Optimization.OptimizationProblem(optf, p_init)
res_ms_fd = Optimization.solve(optprob,
                                ADAM(0.05), maxiters = 300)

# Calculate single shooting loss with parameter from multiple_shoot training
loss_ms_fd, _ = loss_single_shooting(res_ms_fd.minimizer)
println("Multiple shooting loss with ForwardDiffSensitivity: $(loss_ms_fd)")
@test loss_ms_fd < 10loss_ss

# Integration return codes `!= :Success` should return infinite loss.
# In this case, we trigger `retcode = :MaxIters` by setting the solver option `maxiters=1`.
loss_fail, _ = multiple_shoot(p_init, ode_data, tsteps, prob_node, loss_function, Tsit5(),
                              datasize; maxiters=1, verbose=false)
@test loss_fail == Inf

## Test for DomainErrors
@test_throws DomainError multiple_shoot(p_init, ode_data, tsteps, prob_node,
                                        loss_function, Tsit5(), 1)
@test_throws DomainError multiple_shoot(p_init, ode_data, tsteps, prob_node,
                                        loss_function, Tsit5(), datasize + 1)

## Ensembles
u0s = [Float32[2.0, 0.0], Float32[3.0, 1.0]]
function prob_func(prob, i, repeat)
    remake(prob, u0 = u0s[i])
end
ensemble_prob = EnsembleProblem(prob_node, prob_func = prob_func)
ensemble_prob_trueODE = EnsembleProblem(prob_trueode, prob_func = prob_func)
ensemble_alg = EnsembleThreads()
trajectories = 2
ode_data_ensemble = Array(solve(ensemble_prob_trueODE, Tsit5(), ensemble_alg, trajectories = trajectories, saveat = tsteps))

group_size = 3
continuity_term = 200
function loss_multiple_shooting_ens(p)
    return multiple_shoot(p, ode_data_ensemble, tsteps, ensemble_prob, ensemble_alg,
                          loss_function, Tsit5(),
                          group_size; continuity_term,
                          trajectories,
                          abstol=1e-8, reltol=1e-6) # test solver kwargs
end
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((p)->loss_multiple_shooting_ens(p), adtype)
optprob = Optimization.OptimizationProblem(optf, p_init)
res_ms_ensembles = Optimization.solve(optprob,
                                ADAM(0.05), maxiters = 300)

loss_ms_ensembles, _ = loss_single_shooting(res_ms_ensembles.minimizer)

println("Multiple shooting loss with EnsembleProblem: $(loss_ms_ensembles)")

@test loss_ms_ensembles < 10loss_ss
