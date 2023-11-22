"""
    HamiltonianNN(model; ad = AutoForwardDiff())

Constructs a Hamiltonian Neural Network [1]. This neural network is useful for learning
symmetries and conservation laws by supervision on the gradients of the trajectories. It
takes as input a concatenated vector of length `2n` containing the position (of size `n`)
and momentum (of size `n`) of the particles. It then returns the time derivatives for
position and momentum.

!!! note
    This doesn't solve the Hamiltonian Problem. Use [`NeuralHamiltonianDE`](@ref)
    for such applications.

Arguments:

1. `model`: A `Flux.Chain` or `Lux.AbstractExplicitLayer` neural network that returns the
   Hamiltonian of the system.
2. `ad`: The autodiff framework to be used for the internal Hamiltonian computation. The
   default is `AutoForwardDiff()`

!!! note
    If training with Zygote, ensure that the `chunksize` for `AutoForwardDiff` is set to
    `nothing`.

References:

[1] Greydanus, Samuel, Misko Dzamba, and Jason Yosinski. "Hamiltonian Neural Networks." Advances in Neural Information Processing Systems 32 (2019): 15379-15389.
"""
@concrete struct HamiltonianNN{M <: AbstractExplicitLayer} <:
                 AbstractExplicitContainerLayer{(:model,)}
    model::M
    ad
end

function HamiltonianNN(model; ad = AutoForwardDiff())
    @assert ad isa AutoForwardDiff || ad isa AutoZygote || ad isa AutoEnzyme
    !(model isa AbstractExplicitLayer) && (model = Lux.transform(model))
    return HamiltonianNN(model, ad)
end

function __gradient_with_ps(model, psax, N, x)
    function __gradient_closure(psx)
        x_ = reshape(psx[1:N], size(x))
        ps = ComponentArray(psx[(N + 1):end], psax)
        return sum(model(x_, ps))
    end
end

function __hamiltonian_forward(::AutoForwardDiff{nothing}, model, x, ps::ComponentArray)
    psd = getdata(ps)
    psx = vcat(vec(x), psd)
    N = length(x)
    H = ForwardDiff.gradient(__gradient_with_ps(model, getaxes(ps), N, x), psx)
    return reshape(view(H, 1:N), size(x))
end

function __hamiltonian_forward(::AutoForwardDiff{CS}, model, x, ps) where {CS}
    chunksize = CS === nothing ? ForwardDiff.pickchunksize(length(x)) : CS
    __f = sum ∘ Base.Fix2(model, ps)
    cfg = ForwardDiff.GradientConfig(__f, x, ForwardDiff.Chunk{chunksize}())
    return ForwardDiff.gradient(__f, x, cfg)
end

function __hamiltonian_forward(::AutoZygote, model, x, ps)
    return first(Zygote.gradient(sum ∘ model, x, ps))
end

function (hnn::HamiltonianNN{<:LuxCore.AbstractExplicitLayer})(x, ps, st)
    model = StatefulLuxLayer(hnn.model, nothing, st)
    H = __hamiltonian_forward(hnn.ad, model, x, ps)
    n = size(x, 1) ÷ 2
    return vcat(selectdim(H, 1, (n + 1):(2n)), -selectdim(H, 1, 1:n)), model.st
end

"""
    NeuralHamiltonianDE(model, tspan, args...; kwargs...)

Contructs a Neural Hamiltonian DE Layer for solving Hamiltonian Problems parameterized by a
Neural Network [`HamiltonianNN`](@ref).

Arguments:

- `model`: A Flux.Chain, Lux.AbstractExplicitLayer, or Hamiltonian Neural Network that
  predicts the Hamiltonian of the system.
- `tspan`: The timespan to be solved on.
- `kwargs`: Additional arguments splatted to the ODE solver. See the
  [Common Solver Arguments](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/)
  documentation for more details.
"""
@concrete struct NeuralHamiltonianDE{M <: HamiltonianNN} <: NeuralDELayer
    model::M
    tspan
    args
    kwargs
end

function NeuralHamiltonianDE(model, tspan, args...; ad = AutoForwardDiff(), kwargs...)
    hnn = model isa HamiltonianNN ? model : HamiltonianNN(model; ad)
    return NeuralHamiltonianDE(hnn, tspan, args, kwargs)
end

function (nhde::NeuralHamiltonianDE)(x, ps, st)
    model = StatefulLuxLayer(nhde.model, nothing, st)
    neural_hamiltonian(u, p, t) = model(u, p)
    prob = ODEProblem{false}(neural_hamiltonian, x, nhde.tspan, ps)
    sensealg = InterpolatingAdjoint(; autojacvec = ZygoteVJP())
    return solve(prob, nhde.args...; sensealg, nhde.kwargs...), model.st
end
