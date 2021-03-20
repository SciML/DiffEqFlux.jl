# Multiple Shooting

In Multiple Shooting, the training data is split into overlapping intervals.
The solver is then trained on individual intervals. If the end conditions of any
interval co-incide with the initial conditions of the next immediate interval,
then the joined/combined solution is same as solving on the whole dataset
(without splitting).

To ensure that the overlapping part of two consecutive intervals co-incide,
we add a penalizing term, `continuity_strength * absolute_value_of( prediction
of last point of some group, i - prediction of first point of group i+1 )`, to
the loss.

Note that the `continuity_strength` should have a large positive value to add
high penalities in case the solver predicts discontinuous values.


The following is a working demo, using Multiple Shooting

```julia
using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots

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

function plot_function_for_multiple_shoot(plt, pred, grp_size)
	step = 1
	if(grp_size != 1)
		step = grp_size-1
	end
	if(grp_size == datasize)
		scatter!(plt, tsteps, pred[1][1,:], label = "pred")
	else
		for i in 1:step:datasize-grp_size
			# The term `trunc(Integer,(i-1)/(grp_size-1)+1)` goes from 1, 2, ... , N where N is the total number of groups that can be formed from `ode_data` (In other words, N = trunc(Integer, (datasize-1)/(grp_size-1)))
			scatter!(plt, tsteps[i:i+step], pred[trunc(Integer,(i-1)/step+1)][1,:], label = "grp"*string(trunc(Integer,(i-1)/step+1)))
		end
	end
end

callback = function (p, l, pred, predictions; doplot = true)
  display(l)
  if doplot
	# plot the original data
	plt = scatter(tsteps[1:size(pred,2)], ode_data[1,1:size(pred,2)], label = "data")

	# plot the different predictions for individual shoot
	plot_function_for_multiple_shoot(plt, predictions, grp_size_param)

	# plot a single shooting performance of our multiple shooting training (this is what the solver predicts after the training is done)
	# scatter!(plt, tsteps[1:size(pred,2)], pred[1,:], label = "pred")

    display(plot(plt))
  end
  return false
end

# Define parameters for Multiple Shooting
grp_size_param = 1
loss_multiplier_param = 100

neural_ode_f(u,p,t) = dudt2(u,p)
prob_param = ODEProblem(neural_ode_f, u0, tspan, initial_params(dudt2))

function loss_function_param(ode_data, pred):: Float32
	return sum(abs2, (ode_data .- pred))^2
end

function loss_neuralode(p)
	return multiple_shoot(p, ode_data, tsteps, prob_param, loss_function_param, grp_size_param, loss_multiplier_param)
end

result_neuralode = DiffEqFlux.sciml_train(loss_neuralode, prob_neuralode.p,
                                          ADAM(0.05), cb = callback,
                                          maxiters = 300)
callback(result_neuralode.minimizer,loss_neuralode(result_neuralode.minimizer)...;doplot=true)

result_neuralode_2 = DiffEqFlux.sciml_train(loss_neuralode, result_neuralode.minimizer,
                                          BFGS(), cb = callback,
                                          maxiters = 100, allow_f_increases=true)
callback(result_neuralode_2.minimizer,loss_neuralode(result_neuralode_2.minimizer)...;doplot=true)

```
Here's the plots that we get from above

![pic](https://user-images.githubusercontent.com/58384989/111881194-6de9a480-89d5-11eb-8f21-6481d1e22521.PNG)
The picture on the left shows how well our Neural Network does on a single shoot
after training it through `multiple_shoot`.
The picture on the right shows the predictions of each group (Notice that there
are overlapping points as well. These are the points we are trying to co-incide.)

Here is an output with `grp_size` = 30 (which is same as solving on the whole
interval without splitting also called single shooting)

![pic_single_shoot3](https://user-images.githubusercontent.com/58384989/111843307-f0fff180-8926-11eb-9a06-2731113173bc.PNG)

It is clear from the above picture, a single shoot doesn't perform very well
with the ODE Problem we have and gets stuck in a local minima.
