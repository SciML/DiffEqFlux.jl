# Formerly TensorLayer
Base.@deprecate TensorProductBasisFunction(f, n) Basis.GeneralBasisFunction{:none}(f, n, 1)

for B in (:Chebyshev, :Sin, :Cos, :Fourier, :Legendre, :Polynomial)
    Bold = Symbol(B, :Basis)
    @eval Base.@deprecate $(Bold)(n) Basis.$(B)(n)
end

Base.@deprecate TensorLayer(model, out_dim::Int, init_p::F = randn) where {F <: Function} Boltz.Layers.TensorProductLayer(
    model, out_dim; init_weight = init_p)
