# Handling Exogenous Input Signals

The key to using exogeneous input signals is the same as in the rest of the
SciML universe: just use the function in the definition of the differential
equation. For example, if it's a standard differential equation, you can
use the form

```julia
I(t) = t^2

function f(du,u,p,t)
  du[1] = I(t)
  du[2] = u[1]
end
```

so that `I(t)` is an exogenous input signal into `f`. Another form that could be
useful is a closure. For example:

```julia
function f(du,u,p,t,I)
  du[1] = I(t)
  du[2] = u[1]
end

_f = (du,u,p,t) = f(du,u,p,t,x -> x^2)
```

which encloses an extra argument into `f` so that `_f` is now the interface-compliant
differential equation definition.

Note that you can also learn what the exogenous equation is from data. For an
example on how to do this, you can use the [Optimal Control Example](@ref optcontrol)
which shows how to parameterize a `u(t)` by a universal function and learn that
from data.

## Example of a Neural ODE with Exogenous Input

In the following example, a discrete exogenous input signal `ex` is defined and
used as an input into the neural network of a neural ODE system.

```julia
using DifferentialEquations, Flux, Optim, DiffEqFlux, DiffEqSensitivity, Plots

tspan = (0.1f0, Float32(10.0))
tsteps = range(tspan[1], tspan[2], length = 100)
t_vec = collect(tsteps)
ex = vec(ones(Float32,length(tsteps), 1))
f(x) = (atan(8.0 * x - 4.0) + atan(4.0)) / (2.0 * atan(4.0))

function hammerstein_system(u)
    y= zeros(size(u))
    for k in 2:length(u)
        y[k] = 0.2 * f(u[k-1]) + 0.8 * y[k-1]
    end
    return y
end

y = Float32.(hammerstein_system(ex))
plot(collect(tsteps), y, ticks=:native)

nn_model = FastChain(FastDense(2,8, tanh), FastDense(8, 1))
p_model = initial_params(nn_model)

u0 = Float32.([0.0])

function dudt(u, p, t)
    #input_val = u_vals[Int(round(t*10)+1)]
    nn_model(vcat(u[1], ex[Int(round(10*0.1))]), p)
end

prob = ODEProblem(dudt,u0,tspan,nothing)

function predict_neuralode(p)
    _prob = remake(prob,p=p)
    Array(solve(_prob, Tsit5(), saveat=tsteps, abstol = 1e-8, reltol = 1e-6))
end

function loss(p)
    sol = predict_neuralode(p)
    N = length(sol)
    return sum(abs2.(y[1:N] .- sol'))/N
end

# start optimization (I played around with several different optimizers with no success)
res0 = DiffEqFlux.sciml_train(loss,p_model ,ADAM(0.01), maxiters=100)
res1 = DiffEqFlux.sciml_train(loss,res0.minimizer,BFGS(initial_stepnorm=0.01), maxiters=300, allow_f_increases = true)

Flux.gradient(loss,res1.minimizer)

sol = predict_neuralode(res1.minimizer)
plot(tsteps,sol')
N = length(sol)
scatter!(tsteps,y[1:N])

savefig("trained.png")
```

![](https://aws1.discourse-cdn.com/business5/uploads/julialang/original/3X/f/3/f3c2727af36ac20e114fe3c9798e567cc9d22b9e.png)
