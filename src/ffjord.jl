"""
Constructs a continuous-time recurrant neural network, also known as a neural
ordinary differential equation (neural ODE), with a fast gradient calculation
via adjoints [1] and specialized for density estimation based on continuous
normalizing flows (CNF) [2]. At a high level this corresponds to the following steps:

1. Parametrize the variable of interest x(t) as a function f(z,θ,t) of a base variable z(t) with known density p_z;
2. Use the transformation of variables to predict the density p_x as a function
of the density p_z and the trace of the Jacobian of f;
3. Choose the parameter θ to minimize a loss function of p_x (usually the negative likelihood
of the data);

After these steps one may the NN model and the learned θ to predict the density p_z for new
values of z.

```julia
NeuralODE(model,basedist,monte_carlo,tspan,args...;kwargs...)
```
Arguments:
- `model`: A Chain neural network that defines the ̇x.
- `basedist`: Distribution of the base variable. Set to the unit normal by default.
- `monte_carlo`: Method for calcuating the trace of the Jacobian. The default monte_carlo = false
calculates the Jacobian and its trace directly. monte_carlo = true uses the stochastic approach
presented in [3] to provide an unbiased estimate for the trace.
- `tspan`: The timespan to be solved on.
- `kwargs`: Additional arguments splatted to the ODE solver. See the
  [Common Solver Arguments](https://docs.sciml.ai/dev/basics/common_solver_opts/)
  documentation for more details.
Ref
[1]L. S. Pontryagin, Mathematical Theory of Optimal Processes. CRC Press, 1987.
[2]R. T. Q. Chen, Y. Rubanova, J. Bettencourt, D. Duvenaud.
Neural Ordinary Differential Equations. arXiv preprint at arXiv1806.07366, 2019.
[3]W. Grathwohl, R. T. Q. Chen, J. Bettencourt, I. Sutskever, D. Duvenaud.
FFJORD: Free-Form Continuous Dynamic For Scalable Reversible Generative Models.
arXiv preprint at ar1810.01367, 2018.

"""

abstract type CNFLayer <: Function end
Flux.trainable(m::CNFLayer) = (m.p,)

struct FFJORD{M,P,RE,Distribution,Bool,T,A,K} <: CNFLayer
    model::M
    p::P
    re::RE
    basedist::Distribution
    monte_carlo::Bool
    tspan::T
    args::A
    kwargs::K

    function FFJORD(model,tspan,args...;p = nothing,basedist=nothing,monte_carlo=false,kwargs...)
        _p,re = Flux.destructure(model)
        if p === nothing
            p = _p
        end
        if basedist === nothing
            size_input = size(model[1].W)[2]
            basedist = MvNormal(zeros(size_input), I + zeros(size_input,size_input))
        end
        new{typeof(model),typeof(p),typeof(re),typeof(basedist),
            typeof(monte_carlo),typeof(tspan),typeof(args),typeof(kwargs)}(
            model,p,re,basedist,monte_carlo,tspan,args,kwargs)
    end
end

function jacobian_fn(f, x::AbstractVector)
   y::AbstractVector, back = Zygote.pullback(f, x)
   ȳ(i) = [i == j for j = 1:length(y)]
   vcat([transpose(back(ȳ(i))[1]) for i = 1:length(y)]...)
 end

function ffjord(du,u,p,t,re,monte_carlo,e)
    z = @view u[1:end-1]
    m = re(p)
    if monte_carlo
        _, back = Zygote.pullback(m,z)
        eJ = back(e)[1]
        trace_jac = (eJ.*e)[1]
    else
        J = jacobian_fn(m, z)
        trace_jac = length(z) == 1 ? sum(J) : tr(J)
    end
    du[1:end-1] = m(z)
    du[end] = -trace_jac
end

function (n::FFJORD)(x,p=n.p,monte_carlo=n.monte_carlo)
    e = monte_carlo ? randn(Float32,length(x)) : nothing
    ffjord_ = (du,u,p,t)->ffjord(du,u,p,t,n.re,monte_carlo,e)
    prob = ODEProblem{true}(ffjord_,vcat(x,0f0),n.tspan,p)
    pred = solve(prob,Tsit5(),n.args...;n.kwargs...)[:,end]
    pz = n.basedist
    z = pred[1:end-1]
    delta_logp = pred[end]
    logpz = log.(pdf(pz, z))
    logpx = logpz .- delta_logp
    return logpx[1]
end
