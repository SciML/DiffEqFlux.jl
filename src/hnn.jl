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
    To compute the gradients for this layer, it is recommended to use ForwardDiff.jl

To obtain the gradients to train this network, ForwardDiff.gradient is supposed to
be used. Follow this
[tutorial](https://docs.sciml.ai/DiffEqFlux/stable/examples/hamiltonian_nn/) to see how
to define a training loop to circumvent this issue.

```julia
HamiltonianNN(model; p = nothing)
```

Arguments:
1. `model`: A Flux.Chain or Lux.AbstractExplicitLayer neural network that returns the Hamiltonian of the
            system.
2. `p`: The initial parameters of the neural network.

References:

[1] Greydanus, Samuel, Misko Dzamba, and Jason Yosinski. "Hamiltonian Neural Networks." Advances in Neural Information Processing Systems 32 (2019): 15379-15389.

"""
struct HamiltonianNN{M,R,P} <: LuxCore.AbstractExplicitContainerLayer{(:model,)}
    model::M
    re::R
    p::P
    st::NamedTuple

    function HamiltonianNN(model::LuxCore.AbstractExplicitLayer; p=nothing, st=NamedTuple())
        re = nothing
        return new{typeof(model),typeof(re),typeof(p)}(model, re, p, st)
    end
end

function (hnn::HamiltonianNN{<:LuxCore.AbstractExplicitLayer})(x, ps=hnn.p, st=hnn.st)
    (_, st), pb_f = Zygote.pullback(x) do x
        y, st_ = hnn.model(x, ps, st)
        return sum(y), st_
    end
    H = only(pb_f((one(eltype(x)), nothing)))
    n = size(x, 1) ÷ 2
    return vcat(selectdim(H, 1, (n+1):2n), -selectdim(H, 1, 1:n)), st
end

"""
Contructs a Neural Hamiltonian DE Layer for solving Hamiltonian Problems
parameterized by a Neural Network [`HamiltonianNN`](@ref).

```julia
NeuralHamiltonianDE(model, tspan, args...; kwargs...)
```

Arguments:

- `model`: A Flux.Chain, Lux.AbstractExplicitLayer, or Hamiltonian Neural Network that predicts the
           Hamiltonian of the system.
- `tspan`: The timespan to be solved on.
- `kwargs`: Additional arguments splatted to the ODE solver. See the
            [Common Solver Arguments](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/)
            documentation for more details.
"""
struct NeuralHamiltonianDE{M,P,RE,T,A,K} <: NeuralDELayer
    model::HamiltonianNN{M,RE,P}
    p::P
    st::NamedTuple
    tspan::T
    args::A
    kwargs::K
end

function NeuralHamiltonianDE(hnn::HamiltonianNN{M,RE,P}, tspan, args...;
    p=hnn.p, st=hnn.st, kwargs...) where {M,RE,P}
    NeuralHamiltonianDE{M,P,RE,typeof(tspan),typeof(args),
        typeof(kwargs)}(hnn, p, st, tspan, args, kwargs)
end

# TODO: Make sensealg an argument
function NeuralHamiltonianDE(model, tspan, args...; p=nothing, st=NamedTuple(), kwargs...)
    hnn = HamiltonianNN(model, p=p, st=st)
    NeuralHamiltonianDE(hnn, tspan, args...; p, st, kwargs...)
end

function (nhde::NeuralHamiltonianDE{<:LuxCore.AbstractExplicitLayer})(x, ps=nhde.p, st=nhde.st)
    function neural_hamiltonian!(du, u, p, t)
        y, st = nhde.model(u, p, st)
        du .= reshape(y, size(du))
    end
    prob = ODEProblem(ODEFunction{true}(neural_hamiltonian!), x, nhde.tspan, ps)
    # NOTE: Nesting Zygote is an issue. So we can't use ZygoteVJP. Instead we use
    #       ForwardDiff.jl internally.
    sensealg = InterpolatingAdjoint(; autojacvec=true)
    return solve(prob, nhde.args...; sensealg, nhde.kwargs...), st
end
