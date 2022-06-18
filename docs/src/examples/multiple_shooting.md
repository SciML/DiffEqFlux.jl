# Multiple Shooting

In Multiple Shooting, the training data is split into overlapping intervals.
The solver is then trained on individual intervals. If the end conditions of any
interval coincide with the initial conditions of the next immediate interval,
then the joined/combined solution is same as solving on the whole dataset
(without splitting).

To ensure that the overlapping part of two consecutive intervals coincide,
we add a penalizing term, `continuity_term * absolute_value_of(prediction
of last point of group i - prediction of first point of group i+1)`, to
the loss.

Note that the `continuity_term` should have a large positive value to add
high penalties in case the solver predicts discontinuous values.


The following is a working demo, using Multiple Shooting

```@example multipleshoot
using Lux, DiffEqFlux, Optimization, OptimizationPolyalgorithms, DifferentialEquations, Plots
using DiffEqFlux: group_ranges

using Random
rng = Random.default_rng()

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
nn = Lux.Chain(ActivationFunction(x -> x.^3),
                  Lux.Dense(2, 16, tanh),
                  Lux.Dense(16, 2))
p_init, st = Lux.setup(rng, nn)

neuralode = NeuralODE(nn, tspan, Tsit5(), saveat = tsteps)
prob_node = ODEProblem((u,p,t)->nn(u,p,st)[1], u0, tspan, Lux.ComponentArray(p_init))


function plot_multiple_shoot(plt, preds, group_size)
	step = group_size-1
	ranges = group_ranges(datasize, group_size)

	for (i, rg) in enumerate(ranges)
		plot!(plt, tsteps[rg], preds[i][1,:], markershape=:circle, label="Group $(i)")
	end
end

# Animate training
anim = Plots.Animation()
iter = 0
callback = function (p, l, preds; doplot = true)
  display(l)
  global iter
  iter += 1
  if doplot && iter%%10 == 0
    # plot the original data
    plt = scatter(tsteps, ode_data[1,:], label = "Data")

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
	return sum(abs2, data - pred)
end

function loss_multiple_shooting(p)
    return multiple_shoot(p, ode_data, tsteps, prob_node, loss_function, Tsit5(),
                          group_size; continuity_term)
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> loss_multiple_shooting(x), adtype)
optprob = Optimization.OptimizationProblem(optf, Lux.ComponentArray(p_init))
res_ms = Optimization.solve(optprob, PolyOpt(),
                                callback = callback)
gif(anim, "multiple_shooting.gif", fps=5)

```
Here's the animation that we get from above

![pic](https://camo.githubusercontent.com/9f1a4b38895ebaa47b7d90e53268e6f10d04da684b58549624c637e85c22d27b/68747470733a2f2f692e696d6775722e636f6d2f636d507a716a722e676966)
The connected lines show the predictions of each group (Notice that there
are overlapping points as well. These are the points we are trying to coincide.)

Here is an output with `group_size = 30` (which is same as solving on the whole
interval without splitting also called single shooting)

![pic_single_shoot3](https://user-images.githubusercontent.com/58384989/111843307-f0fff180-8926-11eb-9a06-2731113173bc.PNG)

It is clear from the above picture, a single shoot doesn't perform very well
with the ODE Problem we have and gets stuck in a local minima.
