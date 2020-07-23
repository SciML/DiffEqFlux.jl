struct HamiltonianNN{M, R, P}
    model::M
    re::R
    p::P

    function HamiltonianNN(model; p = nothing)
        _p, re = Flux.destructure(model)
        if p === nothing
            p = _p
        end
        return new{typeof(model), typeof(re), typeof(p)}(model, re, p)
    end

    function HamiltonianNN(model::FastChain; p = initial_params(model))
        re = nothing
        return new{typeof(model), typeof(re), typeof(p)}(model, re, p)
    end
end

Flux.trainable(hnn::HamiltonianNN) = (hnn.p,)

function _hamiltonian_forward(re, p, x)
    H = Flux.gradient(x -> sum(re(p)(x)), x)[1]
    n = size(x, 1) ÷ 2
    return cat(H[(n + 1):2n, :], -H[1:n, :], dims=1)
end

function _hamiltonian_forward(m::FastChain, p, x)
    H = Flux.gradient(x -> sum(m(x, p), x))[1]
    n = size(x, 1) ÷ 2
    return cat(H[(n + 1):2n, :], -H[1:n, :], dims=1)
end

(hnn::HamiltonianNN)(x, p = hnn.p) = _hamiltonian_forward(hnn.re, p, x)

(hnn::HamiltonianNN{M})(x, p = hnn.p) where {M<:FastChain} =
    _hamiltonian_forward(hnn.model, p, x)


struct NeuralHamiltonianDE{M,P,RE,T,A,K} <: NeuralDELayer
    hnn::HamiltonianNN{M,RE,P}
    p::P
    tspan::T
    args::A
    kwargs::K

    function NeuralHamiltonianDE(model, tspan, args...; p = nothing, kwargs...)
        hnn = HamiltonianNN(model, p=p)
        new{typeof(hnn.model), typeof(hnn.p), typeof(hnn.re),
            typeof(tspan), typeof(args), typeof(kwargs)}(
            hnn, hnn.p, tspan, args, kwargs)
    end

    function NeuralHamiltonianDE(hnn::HamiltonianNN{M,RE,P}, tspan, args...;
                                 p = hnn.p, kwargs...) where {M,RE,P}
        new{M, P, RE, typeof(tspan), typeof(args),
            typeof(kwargs)}(hnn, hnn.p, tspan, args, kwargs)
    end
end

function (nhde::NeuralHamiltonianDE)(x, p = nhde.p)
    function neural_hamiltonian!(du, u, p, t)
        du .= reshape(nhde.hnn(u, p), size(du))
    end
    prob = ODEProblem(neural_hamiltonian!, x, nhde.tspan, p)
    # NOTE: Nesting Zygote is an issue. So we can't use ZygoteVJP
    sense = InterpolatingAdjoint(autojacvec = false)
    solve(prob, nhde.args...; sensealg = sense, nhde.kwargs...)
end