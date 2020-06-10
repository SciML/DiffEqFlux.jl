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
<<<<<<< HEAD
   y::AbstractVector, back = Zygote.pullback(f, x)
   ȳ(i) = [i == j for j = 1:length(y)]
   vcat([transpose(back(ȳ(i))[1]) for i = 1:length(y)]...)
 end
=======
  y::AbstractVector, back = Zygote.pullback(f, x)
  ȳ(i) = [i == j for j = 1:length(y)]
  vcat([transpose(back(ȳ(i))[1]) for i = 1:length(y)]...)
end
>>>>>>> FFJORD_jacobian

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
