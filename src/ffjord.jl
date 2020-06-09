abstract type CNFLayer <: Function end
Flux.trainable(m::CNFLayer) = (m.p,)

struct FFJORD{M,P,RE,ContinuousUnivariateDistribution,Bool,T,A,K} <: CNFLayer
    model::M
    p::P
    re::RE
    basedist::Distribution
    monte_carlo::Bool
    tspan::T
    args::A
    kwargs::K

    function FFJORD(model,tspan,args...;p = nothing,basedist= Normal(0,1),monte_carlo=false,kwargs...)
        _p,re = Flux.destructure(model)
        if p === nothing
            p = _p
        end
        new{typeof(model),typeof(p),typeof(re),typeof(basedist),
            typeof(monte_carlo),typeof(tspan),typeof(args),typeof(kwargs)}(
            model,p,re,basedist,monte_carlo,tspan,args,kwargs)
    end
end

function ffjord(du,u,p,t,re,monte_carlo,e)
    z = @view u[1:end-1]
    m = re(p)
    fz, back = Zygote.pullback(m,z')
    if monte_carlo
        eJ = back(e)[1]
        jac = -(eJ.*e)[1]
    else
        jac = -sum(back(1)[1])
    end
    du[1:end-1] .= m(z)
    du[end] = jac
end

function (n::FFJORD)(x,p=n.p,monte_carlo=n.monte_carlo)
    e = monte_carlo ? randn(Float32,size(p)[1]) : nothing
    ffjord_ = (du,u,p,t)->ffjord(du,u,p,t,n.re,monte_carlo,e)
    prob = ODEProblem{true}(ffjord_,nothing,n.tspan,nothing)
    pred = concrete_solve(prob,Tsit5(),[x,0f0],p,n.args...;n.kwargs...)[:,end]
    pz = n.basedist
    z = pred[1]
    delta_logp = pred[2]
    logpz = logpdf.(pz, z)
    logpx = logpz - delta_logp
    return logpx
end
