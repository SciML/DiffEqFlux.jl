abstract type CNFLayer <: Function end
Flux.trainable(m::CNFLayer) = (m.p,)

struct FFJORD{M,P,RE,ContinuousUnivariateDistribution,Bool,T,A,K} <: CNFLayer
    model::M
    p::P
    re::RE
    baseDist::ContinuousUnivariateDistribution
    MonteCarlo::Bool
    tspan::T
    args::A
    kwargs::K

    function FFJORD(model,tspan,args...;p = nothing,baseDist=nothing,MonteCarlo=nothing,kwargs...)
        _p,re = Flux.destructure(model)
        if p === nothing
            p = _p
        end
        if baseDist === nothing
            baseDist = Normal(0,1)
        end
        if MonteCarlo === nothing
            MonteCarlo = false
        end
        new{typeof(model),typeof(p),typeof(re),typeof(baseDist),
            typeof(MonteCarlo),typeof(tspan),typeof(args),typeof(kwargs)}(
            model,p,re,baseDist,MonteCarlo,tspan,args,kwargs)
    end
end

function ffjord(du,u,p,t,re,MonteCarlo,e)
    z = @view u[1:end-1]
    m = re(p)
    fz, back = Zygote.pullback(m,z')
    if MonteCarlo
        eJ = back(e)[1]
        jac = -(eJ.*e)[1]
    else
        jac = -sum(back(1)[1])
    end
    du[1:end-1] .= m(z)
    du[end] = jac
end

function (n::FFJORD)(x,p=n.p,MonteCarlo=n.MonteCarlo)
    e = MonteCarlo ? randn(Float32,size(p)[1]) : nothing
    ffjord_ = (du,u,p,t)->ffjord(du,u,p,t,n.re,MonteCarlo,e)
    prob = ODEProblem{true}(ffjord_,nothing,n.tspan,nothing)
    pred = concrete_solve(prob,Tsit5(),[x,0f0],p,n.args...;n.kwargs...)[:,end]
    pz = n.baseDist
    z = pred[1]
    delta_logp = pred[2]
    logpz = logpdf(pz, z)
    logpx = logpz - delta_logp
    return logpx
end
