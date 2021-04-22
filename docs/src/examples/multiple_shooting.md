# Multiple Shooting

In Multiple Shooting, the training data is split into overlapping intervals.
The solver is then trained on individual intervals. If the end conditions of any
interval coincide with the initial conditions of the next immediate interval,
then the joined/combined solution is same as solving on the whole dataset
(without splitting).

To ensure that the overlapping part of two consecutive intervals coincide,
we add a penalizing term, `continuity_term * absolute_value_of( prediction
of last point of some group, i - prediction of first point of group i+1 )`, to
the loss.

Note that the `continuity_strength` should have a large positive value to add
high penalties in case the solver predicts discontinuous values.


The following is a working demo, using Multiple Shooting

```julia
using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots
using DiffEqFlux: group_ranges

# Define initial conditions and time steps
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


function plot_multiple_shoot(plt, preds, group_size)
	step = group_size-1
	ranges = group_ranges(datasize, group_size)

	for (i, rg) in enumerate(ranges)
		plot!(plt, tsteps[rg], preds[i][1,:], markershape=:circle, label="Group $(i)")
	end
end

# Animate training
anim = Animation()
callback = function (p, l, preds; doplot = true)
  display(l)
  if doplot
	# plot the original data
	plt = scatter(tsteps[:], ode_data[1,:], label = "Data")

	# plot the different predictions for individual shoot
	plot_multiple_shoot(plt, preds, group_size)

    frame(anim)
    display(plot(plt))
  end
  return false
end

# Define parameters for Multiple Shooting
group_size = 3
continuity_term = 200

function loss_function(data, pred)
	return sum(abs2, data - pred)^2
end

function loss_multiple_shooting(p)
    return multiple_shoot(p, ode_data, tsteps, prob_node, loss_function, Tsit5(),
                          group_size; continuity_term)
end

res_ms = DiffEqFlux.sciml_train(loss_multiple_shooting, p_init,
                                ADAM(0.05), cb = callback, maxiters = 300)

res_ms = DiffEqFlux.sciml_train(loss_multiple_shooting, res_ms.minimizer,
                                BFGS(), cb = callback, maxiters = 100,
                                allow_f_increases=true)

# gif(anim, "multiple_shooting.gif", fps=15)

```
Here's the plots that we get from above

![pic](https://user-images.githubusercontent.com/58384989/111881194-6de9a480-89d5-11eb-8f21-6481d1e22521.PNG)
The picture on the left shows how well our Neural Network does on a single shoot
after training it through `multiple_shoot`.
The picture on the right shows the predictions of each group (Notice that there
are overlapping points as well. These are the points we are trying to coincide.)

Here is an output with `group_size` = 30 (which is same as solving on the whole
interval without splitting also called single shooting)

![pic_single_shoot3](https://user-images.githubusercontent.com/58384989/111843307-f0fff180-8926-11eb-9a06-2731113173bc.PNG)

It is clear from the above picture, a single shoot doesn't perform very well
with the ODE Problem we have and gets stuck in a local minima.
