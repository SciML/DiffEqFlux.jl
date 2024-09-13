# Tensor Layer
Base.@deprecate TensorProductBasisFunction(f, n) Basis.GeneralBasisFunction{:none}(f, n, 1)

for B in (:Chebyshev, :Sin, :Cos, :Fourier, :Legendre, :Polynomial)
    Bold = Symbol(B, :Basis)
    @eval Base.@deprecate $(Bold)(n) Basis.$(B)(n)
end

Base.@deprecate TensorLayer(model, out_dim::Int, init_p::F = randn) where {F <: Function} Boltz.Layers.TensorProductLayer(
    model, out_dim; init_weight = init_p)

# Spline Layer
function SplineLayer(tspan, tstep, spline_basis; init_saved_points::F = nothing) where {F}
    Base.depwarn(
        "SplineLayer is deprecated and will be removed in the next major release. Refer to \
         Boltz.jl `Layers.SplineLayer` for the newer version.",
        :SplineLayer)

    init_saved_points_corrected = if init_saved_points === nothing
        nothing
    else
        let init_saved_points = init_saved_points
            (rng, _, grid_min, grid_max, grid_step) -> begin
                return init_saved_points(rng, (grid_min, grid_max), grid_step)
            end
        end
    end

    return Layers.SplineLayer((), first(tspan), last(tspan), tstep, spline_basis;
        init_saved_points = init_saved_points_corrected)
end

export SplineLayer

# Hamiltonian Neural Network
Base.@deprecate HamiltonianNN(model; ad = AutoZygote()) Layers.HamiltonianNN{true}(
    model; autodiff = ad)

function NeuralHamiltonianDE(model, tspan, args...; ad = AutoForwardDiff(), kwargs...)
    Base.depwarn(
        "NeuralHamiltonianDE is deprecated, use `NeuralODE` with `Layers.HamiltonianNN` instead.",
        :NeuralHamiltonianDE)
    hnn = model isa Layers.HamiltonianNN ? model : HamiltonianNN(model; ad)
    return NeuralODE(hnn, tspan, args, kwargs)
end

export NeuralHamiltonianDE
