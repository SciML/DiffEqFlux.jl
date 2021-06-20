# [Solving Optimal Control Problems with Universal Differential Equations](@id optcontrol)

Here we will solve a classic optimal control problem with a universal differential
equation. Let

```math
x^{′′} = u^3(t)
```

where we want to optimize our controller `u(t)` such that the following is
minimized:

```math
L(\theta) = \sum_i \Vert 4 - x(t_i) \Vert + 2 \Vert x^\prime(t_i) \Vert + \Vert u(t_i) \Vert
```

where ``i`` is measured on (0,8) at 0.01 intervals. To do this, we rewrite the
ODE in first order form:

```math
\begin{aligned}
x^\prime &= v \\
v^′ &= u^3(t) \\
\end{aligned}
```

and thus

```math
L(\theta) = \sum_i \Vert 4 - x(t_i) \Vert + 2 \Vert v(t_i) \Vert + \Vert u(t_i) \Vert
```

is our loss function on the first order system. We thus choose a neural network
form for ``u`` and optimize the equation with respect to this loss. Note that we
will first reduce control cost (the last term) by 10x in order to bump the network out
of a local minimum. This looks like:

```julia
using DiffEqFlux, DifferentialEquations, Plots, Statistics
tspan = (0.0f0,8.0f0)
ann = FastChain(FastDense(1,32,tanh), FastDense(32,32,tanh), FastDense(32,1))
θ = initial_params(ann)
function dxdt_(dx,x,p,t)
    x1, x2 = x
    dx[1] = x[2]
    dx[2] = ann([t],p)[1]^3
end
x0 = [-4f0,0f0]
ts = Float32.(collect(0.0:0.01:tspan[2]))
prob = ODEProblem(dxdt_,x0,tspan,θ)
solve(prob,Vern9(),abstol=1e-10,reltol=1e-10)
function predict_adjoint(θ)
  Array(solve(prob,Vern9(),p=θ,saveat=ts,sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))))
end
function loss_adjoint(θ)
  x = predict_adjoint(θ)
  mean(abs2,4.0 .- x[1,:]) + 2mean(abs2,x[2,:]) + mean(abs2,[first(ann([t],θ)) for t in ts])/10
end
l = loss_adjoint(θ)
cb = function (θ,l)
  println(l)
  p = plot(solve(remake(prob,p=θ),Tsit5(),saveat=0.01),ylim=(-6,6),lw=3)
  plot!(p,ts,[first(ann([t],θ)) for t in ts],label="u(t)",lw=3)
  display(p)
  return false
end
# Display the ODE with the current parameter values.
cb(θ,l)
loss1 = loss_adjoint(θ)
res1 = DiffEqFlux.sciml_train(loss_adjoint, θ, ADAM(0.005), cb = cb,maxiters=100)
res2 = DiffEqFlux.sciml_train(loss_adjoint, res1.u,
                              BFGS(initial_stepnorm=0.01), cb = cb,maxiters=100,
                              allow_f_increases = false)
```

Now that the system is in a better behaved part of parameter space, we return to
the original loss function to finish the optimization:

```julia
function loss_adjoint(θ)
  x = predict_adjoint(θ)
  mean(abs2,4.0 .- x[1,:]) + 2mean(abs2,x[2,:]) + mean(abs2,[first(ann([t],θ)) for t in ts])
end

res3 = DiffEqFlux.sciml_train(loss_adjoint, res2.u,
                              BFGS(initial_stepnorm=0.01), cb = cb,maxiters=100,
                              allow_f_increases = false)

l = loss_adjoint(res3.u)
cb(res3.u,l)
p = plot(solve(remake(prob,p=res3.u),Tsit5(),saveat=0.01),ylim=(-6,6),lw=3)
plot!(p,ts,[first(ann([t],res3.u)) for t in ts],label="u(t)",lw=3)
savefig("optimal_control.png")
```

![](https://user-images.githubusercontent.com/1814174/81859169-db65b280-9532-11ea-8394-dbb5efcd4036.png)
