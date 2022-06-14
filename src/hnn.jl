"""
Constructs a Hamiltonian Neural Network [1]. This neural network is useful for
learning symmetries and conservation laws by supervision on the gradients
of the trajectories. It takes as input a concatenated vector of length `2n`
containing the position (of size `n`) and momentum (of size `n`) of the
particles. It then returns the time derivatives for position and momentum.

!!! note
    This doesn't solve the Hamiltonian Problem. Use [`NeuralHamiltonianDE`](@ref)
    for such applications.

!!! note
    This layer currently doesn't support GPU. The support will be added in future
    with some AD fixes.

To obtain the gradients to train this network, ReverseDiff.gradient is supposed to
be used. This prevents the usage of `DiffEqFlux.sciml_train` or `Flux.train`. Follow
this [tutorial](https://diffeqflux.sciml.ai/dev/examples/hamiltonian_nn/) to see how
to define a training loop to circumvent this issue.

```julia
HamiltonianNN(model; p = nothing)
HamiltonianNN(model::Lux.AbstractExplicitLayer; p = nothing)
```

Arguments:
1. `model`: A Flux.Chain or Lux.AbstractExplicitLayer neural network that returns the Hamiltonian of the
            system.
2. `p`: The initial parameters of the neural network.

References:

[1] Greydanus, Samuel, Misko Dzamba, and Jason Yosinski. "Hamiltonian Neural Networks." Advances in Neural Information Processing Systems 32 (2019): 15379-15389.

"""
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

    function HamiltonianNN(model::Lux.Chain; p = nothing)
        return new{typeof(model), typeof(re), typeof(p)}(model, re, p)
    end
end

Flux.trainable(hnn::HamiltonianNN) = (hnn.p,)

function _hamiltonian_forward(re, p, x, args...)
    H = Flux.gradient(x -> sum(re(p)(x)), x)[1]
    n = size(x, 1) ÷ 2
    return cat(H[(n + 1):2n, :], -H[1:n, :], dims=1)
end

function _hamiltonian_forward(re::Lux.Chain, p, x, args...)
    st = args[1]
    H = Lux.gradient(x -> sum(Lux.apply(re,x,p,st)[1]), x)[1]
    n = size(x, 1) ÷ 2
    return cat(H[(n + 1):2n, :], -H[1:n, :], dims=1), st
end

(hnn::HamiltonianNN)(x, p = hnn.p) = _hamiltonian_forward(hnn.re, p, x)
(hnn::HamiltonianNN{M})(x, p, st) where {M<:Lux.AbstractExplicitLayer} = _hamiltonian_forward(hnn.model, p, x, st)


"""
Contructs a Neural Hamiltonian DE Layer for solving Hamiltonian Problems
parameterized by a Neural Network [`HamiltonianNN`](@ref).

```julia
NeuralHamiltonianDE(model, tspan, args...; kwargs...)
```

Arguments:

- `model`: A Chain, Lux.AbstractExplicitLayer or Hamiltonian Neural Network that predicts the
           Hamiltonian of the system.
- `tspan`: The timespan to be solved on.
- `kwargs`: Additional arguments splatted to the ODE solver. See the
            [Common Solver Arguments](https://diffeq.sciml.ai/dev/basics/common_solver_opts/)
            documentation for more details.
"""
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

function (nhde::NeuralHamiltonianDE{M})(x, p, st) where {M<:Lux.AbstractExplicitLayer}
    function neural_hamiltonian!(du, u, p, t)
        du .= reshape(nhde.hnn(u, p, st)[1], size(du))
    end
    prob = ODEProblem(neural_hamiltonian!, x, nhde.tspan, p)
    # NOTE: Nesting Zygote is an issue. So we can't use ZygoteVJP
    sense = InterpolatingAdjoint(autojacvec = false)
    solve(prob, nhde.args...; sensealg = sense, nhde.kwargs...)
end
