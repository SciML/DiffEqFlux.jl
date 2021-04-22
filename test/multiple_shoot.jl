using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Test
using DiffEqFlux: group_ranges

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
nn = FastChain((x, p) -> x.^3,
                FastDense(2, 16, tanh),
                FastDense(16, 2))
p_init = initial_params(nn)

neuralode = NeuralODE(nn, tspan, Tsit5(), saveat = tsteps)
prob_node = ODEProblem((u,p,t)->nn(u,p), u0, tspan, p_init)

function predict_single_shooting(p)
    return Array(neuralode(u0, p))
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

res_single_shooting = DiffEqFlux.sciml_train(loss_single_shooting, neuralode.p,
                                          ADAM(0.05),
										  maxiters = 300)

loss_ss, _ = loss_single_shooting(res_single_shooting.minimizer)
println("Single shooting loss: $(loss_ss)")

## Test Multiple Shooting
group_size = 2
continuity_term = 100

function loss_multiple_shooting(p)
    return multiple_shoot(p, ode_data, tsteps, prob_node, loss_function, Tsit5(),
                          group_size; continuity_term)
end

res_ms = DiffEqFlux.sciml_train(loss_multiple_shooting, neuralode.p,
                                ADAM(0.05), maxiters = 300)

# Calculate single shooting loss with parameter from multiple_shoot training
loss_ms, _ = loss_single_shooting(res_ms.minimizer)
println("Multiple shooting loss: $(loss_ms)")
@test loss_ms < loss_ss

# Test for DomainErrors
@test_throws DomainError multiple_shoot(p_init, ode_data, tsteps, prob_node,
                                        loss_function, Tsit5(), 1)
@test_throws DomainError multiple_shoot(p_init, ode_data, tsteps, prob_node,
                                        loss_function, Tsit5(), datasize + 1)
