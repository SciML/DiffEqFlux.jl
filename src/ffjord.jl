abstract type CNFLayer <: Function end
Flux.trainable(m::CNFLayer) = (m.p,)

struct DeterministicCNFLayer{M,P,RE,Distribution,T,A,K} <: CNFLayer
    model::M
    p::P
    re::RE
    basedist::Distribution
    tspan::T
    args::A
    kwargs::K

    function DeterministicCNFLayer(model,tspan,args...;p = nothing,basedist=nothing,kwargs...)
        _p,re = Flux.destructure(model)
        if p === nothing
            p = _p
        end
        if basedist === nothing
            size_input = size(model[1].W)[2]
            basedist = MvNormal(zeros(size_input), I + zeros(size_input,size_input))
        end
        new{typeof(model),typeof(p),typeof(re),typeof(basedist),typeof(tspan),typeof(args),typeof(kwargs)}(
            model,p,re,basedist,tspan,args,kwargs)
    end
end

"""
Constructs a continuous-time recurrent neural network, also known as a neural
ordinary differential equation (neural ODE), with fast gradient calculation
via adjoints [1] and specialized for density estimation based on continuous
normalizing flows (CNF) [2] with a stochastic approach [2] for the computation of the trace
of the dynamics' jacobian. At a high level this corresponds to the following steps:

1. Parameterize the variable of interest x(t) as a function f(z,θ,t) of a base variable z(t) with known density p_z;
2. Use the transformation of variables formula to predict the density p_x as a function of the density p_z and the trace of the Jacobian of f;
3. Choose the parameter θ to minimize a loss function of p_x (usually the negative likelihood of the data);

After these steps one may use the NN model and the learned θ to predict the density p_x for new values of x.

```julia
FFJORD(model,basedist=nothing,monte_carlo=false,tspan,args...;kwargs...)
```
Arguments:
- `model`: A Chain neural network that defines the ̇x.
- `basedist`: Distribution of the base variable. Set to the unit normal by default.
- `tspan`: The timespan to be solved on.
- `kwargs`: Additional arguments splatted to the ODE solver. See the
  [Common Solver Arguments](https://diffeq.sciml.ai/dev/basics/common_solver_opts/)
  documentation for more details.
Ref
[1]L. S. Pontryagin, Mathematical Theory of Optimal Processes. CRC Press, 1987.
[2]R. T. Q. Chen, Y. Rubanova, J. Bettencourt, D. Duvenaud. Neural Ordinary Differential Equations. arXiv preprint at arXiv1806.07366, 2019.
[3]W. Grathwohl, R. T. Q. Chen, J. Bettencourt, I. Sutskever, D. Duvenaud. FFJORD: Free-Form Continuous Dynamic For Scalable Reversible Generative Models. arXiv preprint at arXiv1810.01367, 2018.

"""
struct FFJORDLayer{M,P,RE,Distribution,T,A,K} <: CNFLayer
    model::M
    p::P
    re::RE
    basedist::Distribution
    tspan::T
    args::A
    kwargs::K

    function FFJORDLayer(model,tspan,args...;p = nothing,basedist=nothing,kwargs...)
        _p,re = Flux.destructure(model)
        if p === nothing
            p = _p
        end
        if basedist === nothing
            size_input = size(model[1].W)[2]
            basedist = MvNormal(zeros(size_input), I + zeros(size_input,size_input))
        end
        new{typeof(model),typeof(p),typeof(re),typeof(basedist),typeof(tspan),typeof(args),typeof(kwargs)}(
            model,p,re,basedist,tspan,args,kwargs)
    end
end

function jacobian_fn(f, x::AbstractVector)
   y::AbstractVector, back = Zygote.pullback(f, x)
   ȳ(i) = [i == j for j = 1:length(y)]
   vcat([transpose(back(ȳ(i))[1]) for i = 1:length(y)]...)
 end

function cnf(du,u,p,t,re)
    z = @view u[1:end-1]
    m = re(p)
    J = jacobian_fn(m, z)
    trace_jac = length(z) == 1 ? sum(J) : tr(J)
    du[1:end-1] = m(z)
    du[end] = -trace_jac
end

function ffjord(du,u,p,t,re,e,monte_carlo)
    z = @view u[1:end-3]
    m = re(p)
    _, back = Zygote.pullback(m,z)
    eJ = back(e)[1]
    if monte_carlo
        trace_jac = (eJ.*e)[1]
    else
        J = jacobian_fn(m, z)
        trace_jac = length(z) == 1 ? sum(J) : tr(J)
    end
    du[1:end-3] = m(z)
    du[end-2] = -trace_jac
    du[end-1] = sum(abs2, m(z))
    du[end] = norm(eJ)^2
end

function (n::DeterministicCNFLayer)(x,p=n.p)
    cnf_ = (du,u,p,t)->cnf(du,u,p,t,n.re)
    prob = ODEProblem{true}(cnf_,vcat(x,0f0),n.tspan,p)
    sense = InterpolatingAdjoint(autojacvec = false)
    pred = solve(prob,n.args...;sensealg=sense,n.kwargs...)[:,end]
    pz = n.basedist
    z = pred[1:end-1]
    delta_logp = pred[end]
    logpz = logpdf(pz, z)
    logpx = logpz .- delta_logp
    return logpx[1]
end

function (n::FFJORDLayer)(x,p=n.p,monte_carlo=true)
    e = randn(Float32,length(x))
    ffjord_ = (du,u,p,t)->ffjord(du,u,p,t,n.re,e,monte_carlo)
    prob = ODEProblem{true}(ffjord_,vcat(x,0f0,0f0,0f0),n.tspan,p)
    sense = InterpolatingAdjoint(autojacvec = false)
    pred = solve(prob,n.args...;sensealg=sense,n.kwargs...)[:,end]
    pz = n.basedist
    z = pred[1:end-3]
    delta_logp = pred[end-2]
    reg1 = pred[end-1]
    reg2 =  pred[end]
    logpz = logpdf(pz, z)
    logpx = logpz .- delta_logp
    return logpx[1], reg1, reg2
end
