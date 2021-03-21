using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Test

# General loss function to compare single shooting and multiple shooting predictions
function general_loss_function(result_neuralode)
	return sum(abs2, (ode_data[:,:] .- Array(prob_neuralode(u0, result_neuralode.minimizer)) ))
end

# Define initial conditions and timesteps
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
dudt2 = FastChain((x, p) -> x.^3,
                  FastDense(2, 16, tanh),
                  FastDense(16, 2))
prob_neuralode = NeuralODE(dudt2, (0.0,5.0), Tsit5(), saveat = tsteps)

function loss_neuralode(p)
    pred = Array(prob_neuralode(u0, p))
    loss = sum(abs2, (ode_data[:,1:size(pred,2)] .- pred))
    return loss, pred
end


result_neuralode = DiffEqFlux.sciml_train(loss_neuralode, prob_neuralode.p,
                                          ADAM(0.05),
										  maxiters = 300)

single_shoot_loss = general_loss_function(result_neuralode)
println("single_shoot_loss: ",single_shoot_loss)

# Define parameters for Multiple Shooting
grp_size_param = 1
loss_multiplier_param = 100

neural_ode_f(u,p,t) = dudt2(u,p)
prob_param = ODEProblem(neural_ode_f, u0, tspan, initial_params(dudt2))

function loss_function_param(ode_data, pred):: Float32
	return sum(abs2, (ode_data .- pred))^2
end

function loss_neuralode_param(p)
	return multiple_shoot(p, ode_data, tsteps, prob_param, loss_function_param, grp_size_param, loss_multiplier_param)
end


multiple_shoot_result_neuralode_1 = DiffEqFlux.sciml_train(loss_neuralode_param, prob_neuralode.p,
                                          ADAM(0.05),
                                          maxiters = 300)

multiple_shoot_loss_1 = general_loss_function(multiple_shoot_result_neuralode_1)
println("multiple_shoot_loss_1: ",multiple_shoot_loss_1)


# test for grp_size = 1
@test multiple_shoot_loss_1 < single_shoot_loss

# test for grp_size = 5
grp_size_param = 5
multiple_shoot_result_neuralode_2 = DiffEqFlux.sciml_train(loss_neuralode_param, prob_neuralode.p,
                                          ADAM(0.05),
                                          maxiters = 300)

multiple_shoot_loss_2 = general_loss_function(multiple_shoot_result_neuralode_2)
println("multiple_shoot_loss_2: ",multiple_shoot_loss_2)

@test multiple_shoot_loss_2 < single_shoot_loss
