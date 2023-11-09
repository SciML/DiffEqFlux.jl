module DiffEqFlux

import PrecompileTools

PrecompileTools.@recompile_invalidations begin
    using ChainRulesCore, ConcreteStructs,
        LinearAlgebra, Lux, LuxCore, Random, Reexport, SciMLBase, SciMLSensitivity, Zygote
end

import ChainRulesCore as CRC
import LuxCore: AbstractExplicitLayer, AbstractExplicitContainerLayer
import Lux.Experimental: StatefulLuxLayer

# using Adapt, Base.Iterators, ChainRulesCore, ConsoleProgressMonitor,
#     DataInterpolations, DiffEqBase, Distributions, DistributionsAD,
#     ForwardDiff, Functors, LinearAlgebra, Logging, LoggingExtras, LuxCore,
#     Printf, ProgressLogging, Random, RecursiveArrayTools, Reexport,
#     SciMLBase, TerminalLoggers, Zygote, ZygoteRules

@reexport using Lux, SciMLSensitivity

# gpu_or_cpu(x) = Array

# # ForwardDiff integration

# ZygoteRules.@adjoint function ForwardDiff.Dual{T}(x, ẋ::Tuple) where {T}
#     @assert length(ẋ) == 1
#     ForwardDiff.Dual{T}(x, ẋ), ḋ -> (ḋ.partials[1], (ḋ.value,))
# end

# ZygoteRules.@adjoint ZygoteRules.literal_getproperty(d::ForwardDiff.Dual{T}, ::Val{:partials}) where {T} =
#     d.partials, ṗ -> (ForwardDiff.Dual{T}(ṗ[1], 0),)

# ZygoteRules.@adjoint ZygoteRules.literal_getproperty(d::ForwardDiff.Dual{T}, ::Val{:value}) where {T} =
#     d.value, ẋ -> (ForwardDiff.Dual{T}(0, ẋ),)

# ZygoteRules.@adjoint ZygoteRules.literal_getproperty(A::Tridiagonal, ::Val{:dl}) = A.dl, y -> Tridiagonal(dl, zeros(length(d)), zeros(length(du)),)
# ZygoteRules.@adjoint ZygoteRules.literal_getproperty(A::Tridiagonal, ::Val{:d}) = A.d, y -> Tridiagonal(zeros(length(dl)), d, zeros(length(du)),)
# ZygoteRules.@adjoint ZygoteRules.literal_getproperty(A::Tridiagonal, ::Val{:du}) = A.dl, y -> Tridiagonal(zeros(length(dl)), zeros(length(d), du),)
# ZygoteRules.@adjoint Tridiagonal(dl, d, du) = Tridiagonal(dl, d, du), p̄ -> (diag(p̄[2:end, 1:end-1]), diag(p̄), diag(p̄[1:end-1, 2:end]))

# FIXME: Type Piracy
function CRC.rrule(::Type{Tridiagonal}, dl, d, du)
    y = Tridiagonal(dl, d, du)
    @views function ∇Tridiagonal(∂y)
        return (NoTangent(), diag(∂y[2:end, 1:(end - 1)]), diag(∂y),
            diag(∂y[1:(end - 1), 2:end]))
    end
    return y, ∇Tridiagonal
end

# include("ffjord.jl")
include("neural_de.jl")
include("spline_layer.jl")
include("tensor_product.jl")
include("collocation.jl")
# include("hnn.jl")
include("multiple_shooting.jl")

export NeuralODE, NeuralDSDE, NeuralSDE, NeuralCDDE, NeuralDAE, AugmentedNDELayer,
    NeuralODEMM, TensorLayer, SplineLayer
# export FFJORD, NeuralHamiltonianDE
# export HamiltonianNN
export TensorProductBasisFunction,
    ChebyshevBasis, SinBasis, CosBasis, FourierBasis, LegendreBasis, PolynomialBasis
# export FFJORDDistribution
export DimMover

export EpanechnikovKernel, UniformKernel, TriangularKernel, QuarticKernel, TriweightKernel,
    TricubeKernel, GaussianKernel, CosineKernel, LogisticKernel, SigmoidKernel,
    SilvermanKernel
export collocate_data

export multiple_shoot

end
