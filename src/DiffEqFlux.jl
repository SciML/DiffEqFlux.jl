module DiffEqFlux

using Adapt, Base.Iterators, ConsoleProgressMonitor, DataInterpolations,
    DiffEqBase, SciMLSensitivity, DiffResults, Distributions, DistributionsAD,
    ForwardDiff, Optimization, OptimizationPolyalgorithms, LinearAlgebra,
    Logging, LoggingExtras, Printf, ProgressLogging, Random, RecursiveArrayTools,
    Reexport, SciMLBase, StaticArrays, TerminalLoggers, Zygote, ZygoteRules

@reexport using SciMLSensitivity
@reexport using Zygote

# deprecate

import OptimizationFlux
import NNlib
import Lux
using Requires
using Cassette
@reexport using Flux
@reexport using OptimizationOptimJL

gpu_or_cpu(x) = Array

# ForwardDiff integration

ZygoteRules.@adjoint function ForwardDiff.Dual{T}(x, ẋ::Tuple) where T
  @assert length(ẋ) == 1
  ForwardDiff.Dual{T}(x, ẋ), ḋ -> (ḋ.partials[1], (ḋ.value,))
end

ZygoteRules.@adjoint ZygoteRules.literal_getproperty(d::ForwardDiff.Dual{T}, ::Val{:partials}) where T =
  d.partials, ṗ -> (ForwardDiff.Dual{T}(ṗ[1], 0),)

ZygoteRules.@adjoint ZygoteRules.literal_getproperty(d::ForwardDiff.Dual{T}, ::Val{:value}) where T =
  d.value, ẋ -> (ForwardDiff.Dual{T}(0, ẋ),)

ZygoteRules.@adjoint ZygoteRules.literal_getproperty(A::Tridiagonal, ::Val{:dl}) = A.dl,y -> Tridiagonal(dl,zeros(length(d)),zeros(length(du)),)
ZygoteRules.@adjoint ZygoteRules.literal_getproperty(A::Tridiagonal, ::Val{:d}) = A.d,y -> Tridiagonal(zeros(length(dl)),d,zeros(length(du)),)
ZygoteRules.@adjoint ZygoteRules.literal_getproperty(A::Tridiagonal, ::Val{:du}) = A.dl,y -> Tridiagonal(zeros(length(dl)),zeros(length(d),du),)
ZygoteRules.@adjoint Tridiagonal(dl, d, du) = Tridiagonal(dl, d, du), p̄ -> (diag(p̄[2:end,1:end-1]),diag(p̄),diag(p̄[1:end-1,2:end]))

include("ffjord.jl")
include("train.jl")
include("fast_layers.jl")
include("neural_de.jl")
include("require.jl")
include("spline_layer.jl")
include("tensor_product_basis.jl")
include("tensor_product_layer.jl")
include("collocation.jl")
include("hnn.jl")
include("multiple_shooting.jl")

Flux.device(::FastLayer) = @warn "device(f::FastLayer) is a no-op: to move FastChain computations to a GPU, apply gpu(x) to the weight vector"
Flux.gpu(::FastLayer) = @warn "device(f::FastLayer) is a no-op: to move FastChain computations to a GPU, apply gpu(x) to the weight vector"
Flux.cpu(::FastLayer) = @warn "device(f::FastLayer) is a no-op: to move FastChain computations to a CPU, apply cpu(x) to the weight vector"

export DeterministicCNF, FFJORD, NeuralODE, NeuralDSDE, NeuralSDE, NeuralCDDE, NeuralDAE, NeuralODEMM, TensorLayer, AugmentedNDELayer, SplineLayer, NeuralHamiltonianDE
export HamiltonianNN
export ChebyshevBasis, SinBasis, CosBasis, FourierBasis, LegendreBasis, PolynomialBasis
export FastDense, StaticDense, FastChain, initial_params
export FFJORDDistribution

export EpanechnikovKernel, UniformKernel, TriangularKernel, QuarticKernel
export TriweightKernel, TricubeKernel, GaussianKernel, CosineKernel
export LogisticKernel, SigmoidKernel, SilvermanKernel
export collocate_data

export multiple_shoot

end
