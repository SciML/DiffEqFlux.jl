# Benchmarks

## Vs Torchdiffeq on small ODEs

A raw ODE solver benchmark showcases [a 30,000x performance advantage over
torchdiffeq on small
ODEs](https://gist.github.com/ChrisRackauckas/cc6ac746e2dfd285c28e0584a2bfd320).

## A bunch of adjoint choices on neural ODEs

```julia
using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots, DiffEqSensitivity, Zygote, BenchmarkTools

u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

dudt2 = FastChain((x, p) -> x.^3,
                  FastDense(2, 50, tanh),
                  FastDense(50, 2))
p = initial_params(dudt2)

prob_neuralode_interpolating = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps, sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))

function loss_neuralode_interpolating(p)
    pred = Array(prob_neuralode_interpolating(u0, p))
    loss = sum(abs2, ode_data .- pred)
    return loss
end

@btime Zygote.gradient(loss_neuralode_interpolating,p)
# 6.845 ms (128551 allocations: 3.24 MiB)

prob_neuralode_interpolating_zygote = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))

function loss_neuralode_interpolating_zygote(p)
    pred = Array(prob_neuralode_interpolating_zygote(u0, p))
    loss = sum(abs2, ode_data .- pred)
    return loss
end

@btime Zygote.gradient(loss_neuralode_interpolating_zygote,p)
# 3.065 ms (62591 allocations: 7.44 MiB)

prob_neuralode_backsolve = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps, sensealg=BacksolveAdjoint(autojacvec=ReverseDiffVJP(true)))

function loss_neuralode_backsolve(p)
    pred = Array(prob_neuralode_backsolve(u0, p))
    loss = sum(abs2, ode_data .- pred)
    return loss
end

@btime Zygote.gradient(loss_neuralode_backsolve,p)
# 4.543 ms (78090 allocations: 2.22 MiB)

prob_neuralode_quad = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps, sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true)))

function loss_neuralode_quad(p)
    pred = Array(prob_neuralode_quad(u0, p))
    loss = sum(abs2, ode_data .- pred)
    return loss
end

@btime Zygote.gradient(loss_neuralode_quad,p)
# 10.040 ms (80512 allocations: 4.06 MiB

prob_neuralode_backsolve_tracker = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps, sensealg=BacksolveAdjoint(autojacvec=TrackerVJP()))

function loss_neuralode_backsolve_tracker(p)
    pred = Array(prob_neuralode_backsolve_tracker(u0, p))
    loss = sum(abs2, ode_data .- pred)
    return loss
end

@btime Zygote.gradient(loss_neuralode_backsolve_tracker,p)
# 21.916 ms (167388 allocations: 11.15 MiB)

prob_neuralode_backsolve_zygote = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps, sensealg=BacksolveAdjoint(autojacvec=ZygoteVJP()))

function loss_neuralode_backsolve_zygote(p)
    pred = Array(prob_neuralode_backsolve_zygote(u0, p))
    loss = sum(abs2, ode_data .- pred)
    return loss
end

@btime Zygote.gradient(loss_neuralode_backsolve_zygote,p)
# 1.807 ms (39676 allocations: 5.09 MiB)

prob_neuralode_backsolve_false = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps, sensealg=BacksolveAdjoint(autojacvec=ReverseDiffVJP(false)))

function loss_neuralode_backsolve_false(p)
    pred = Array(prob_neuralode_backsolve_false(u0, p))
    loss = sum(abs2, ode_data .- pred)
    return loss
end

@btime Zygote.gradient(loss_neuralode_backsolve_false,p)
# 3.905 ms (10311 allocations: 1.18 MiB)

prob_neuralode_tracker = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps, sensealg=TrackerAdjoint())

function loss_neuralode_tracker(p)
    pred = Array(prob_neuralode_tracker(u0, p))
    loss = sum(abs2, ode_data .- pred)
    return loss
end

@btime Zygote.gradient(loss_neuralode_tracker,p)
# 15.187 ms (106345 allocations: 4.48 MiB)
```
