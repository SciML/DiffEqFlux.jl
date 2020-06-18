# Neural Second Order Ordinary Differential Equation

The neural ODE focuses and finding a neural network such that:

```math
u^\prime = NN(u)
```

However, in many cases in physics-based modeling, the key object is not the
velocity but the acceleration: knowing the acceleration tells you the force
field and thus the generating process for the dynamical system. Thus what we want
to do is find the force, i.e.:

```math
u^{\prime\prime} = NN(u)
```

(Note that in order to be the acceleration, we should divide the output of the
neural network by the mass!)

An example of training a neural network on a second order ODE is as follows:

```julia
using OrdinaryDiffEq, Flux, DiffEqFlux, DiffEqSensitivity, Zygote, RecursiveArrayTools

u0 = Float32[0.; 2.]
du0 = Float32[0.; 0.]
tspan = (0.0f0, 1.0f0)
t = range(tspan[1], tspan[2], length=20)

model = FastChain(FastDense(2, 50, tanh), FastDense(50, 2))
p = initial_params(model)
ff(du,u,p,t) = model(u,p)
prob = SecondOrderODEProblem{false}(ff, du0, u0, tspan, p)

function predict(p)
    Array(solve(prob, Tsit5(), p=p, saveat=t))
end

correct_pos = Float32.(transpose(hcat(collect(0:0.05:1)[2:end], collect(2:-0.05:1)[2:end])))

function loss_n_ode(p)
    pred = predict(p)
    sum(abs2, correct_pos .- pred[1:2, :]), pred
end

data = Iterators.repeated((), 1000)
opt = ADAM(0.01)

l1 = loss_n_ode(p)

cb = function (p,l,pred)
    println(l)
    l < 0.01
end

res = DiffEqFlux.sciml_train(loss_n_ode, p, opt, cb=cb, maxiters = 1000)
```
