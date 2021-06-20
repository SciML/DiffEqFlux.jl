# Simultaneous Fitting of Multiple Neural Networks

In many cases users are interested in fitting multiple neural networks
or parameters simultaneously. This tutorial addresses how to perform
this kind of study.

The following is a fully working demo on the Fitzhugh-Nagumo ODE:

```julia
using DiffEqFlux, DifferentialEquations

function fitz(du,u,p,t)
  v,w = u
  a,b,τinv,l = p
  du[1] = v - v^3/3 -w + l
  du[2] = τinv*(v +  a - b*w)
end

p_ = Float32[0.7,0.8,1/12.5,0.5]
u0 = [1f0;1f0]
tspan = (0f0,10f0)
prob = ODEProblem(fitz,u0,tspan,p_)
sol = solve(prob, Tsit5(), saveat = 0.5 )

# Ideal data
X = Array(sol)
Xₙ = X + Float32(1e-3)*randn(eltype(X), size(X))  #noisy data

# For xz term
NN_1 = FastChain(FastDense(2, 16, tanh), FastDense(16, 1))
p1 = initial_params(NN_1)

# for xy term
NN_2 = FastChain(FastDense(3, 16, tanh), FastDense(16, 1))
p2 = initial_params(NN_2)
scaling_factor = 1f0

p = [p1;p2;scaling_factor]
function dudt_(u,p,t)
    v,w = u
    z1 = NN_1([v,w], p[1:length(p1)])
    z2 = NN_2([v,w,t], p[(length(p1)+1):end-1])
    [z1[1],p[end]*z2[1]]
end
prob_nn = ODEProblem(dudt_,u0, tspan, p)
sol_nn = solve(prob_nn, Tsit5(),saveat = sol.t)

function predict(θ)
    Array(solve(prob_nn, Vern7(), p=θ, saveat = sol.t,
                         abstol=1e-6, reltol=1e-6,
                         sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))))
end

# No regularisation right now
function loss(θ)
    pred = predict(θ)
    sum(abs2, Xₙ .- pred), pred
end
loss(p)
const losses = []
callback(θ,l,pred) = begin
    push!(losses, l)
    if length(losses)%50==0
        println(losses[end])
    end
    false
end

res1_uode = DiffEqFlux.sciml_train(loss, p, ADAM(0.01), cb=callback, maxiters = 500)
res2_uode = DiffEqFlux.sciml_train(loss, res1_uode.minimizer, BFGS(initial_stepnorm=0.01), cb=callback, maxiters = 10000)
```

The key is that `sciml_train` acts on a single parameter vector `p`.
Thus what we do here is concatenate all of the parameters into a single
vector `p = [p1;p2;scaling_factor]` and then train on this parameter
vector. Whenever we need to evaluate the neural networks, we cut the
vector and grab the portion that corresponds to the neural network.
For example, the `p1` portion is `p[1:length(p1)]`, which is why the
first neural network's evolution is written like `NN_1([v,w], p[1:length(p1)])`.

This method is flexible to use with many optimizers and in fairly
optimized ways. The allocations can be reduced by using `@view p[1:length(p1)]`.
We can also see with the `scaling_factor` that we can grab parameters
directly out of the vector and use them as needed.
