module DiffEqFlux

using Adapt, Base.Iterators, ConsoleProgressMonitor, DataInterpolations,
    DiffEqBase, Distributions, DistributionsAD,
    ForwardDiff, LinearAlgebra, Lux,
    Logging, LoggingExtras, Printf, ProgressLogging, Random, RecursiveArrayTools,
    Reexport, SciMLBase, TerminalLoggers, Zygote

@reexport using SciMLSensitivity
@reexport using Flux
using Functors

import ChainRulesCore

gpu_or_cpu(x) = Array

include("ffjord.jl")
include("neural_de.jl")
include("spline_layer.jl")
include("tensor_product_basis.jl")
include("tensor_product_layer.jl")
include("collocation.jl")
include("hnn.jl")
include("multiple_shooting.jl")

export FFJORD, NeuralODE, NeuralDSDE, NeuralSDE, NeuralCDDE, NeuralDAE, 
       NeuralODEMM, TensorLayer, AugmentedNDELayer, SplineLayer, NeuralHamiltonianDE
export HamiltonianNN
export ChebyshevBasis, SinBasis, CosBasis, FourierBasis, LegendreBasis, PolynomialBasis
export FastDense, StaticDense, initial_params
export FFJORDDistribution
export DimMover, FluxBatchOrder

export EpanechnikovKernel, UniformKernel, TriangularKernel, QuarticKernel
export TriweightKernel, TricubeKernel, GaussianKernel, CosineKernel
export LogisticKernel, SigmoidKernel, SilvermanKernel
export collocate_data

export multiple_shoot

end
