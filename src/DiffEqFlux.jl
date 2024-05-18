module DiffEqFlux

using PrecompileTools: @recompile_invalidations

@recompile_invalidations begin
    using ADTypes: ADTypes, AutoForwardDiff, AutoZygote, AutoEnzyme
    using ChainRulesCore: ChainRulesCore
    using ComponentArrays: ComponentArray
    using ConcreteStructs: @concrete
    using Distributions: Distributions, ContinuousMultivariateDistribution, Distribution,
                         logpdf
    using DistributionsAD: DistributionsAD
    using ForwardDiff: ForwardDiff
    using Functors: Functors, fmap
    using LinearAlgebra: LinearAlgebra, Diagonal, det, diagind, mul!
    using Lux: Lux, Chain, Dense, StatefulLuxLayer
    using LuxCore: LuxCore, AbstractExplicitLayer, AbstractExplicitContainerLayer
    using Random: Random, AbstractRNG, randn!
    using Reexport: @reexport
    using SciMLBase: SciMLBase, DAEProblem, DDEFunction, DDEProblem, EnsembleProblem,
                     ODEFunction, ODEProblem, ODESolution, SDEFunction, SDEProblem, remake,
                     solve
    using SciMLSensitivity: SciMLSensitivity, AdjointLSS, BacksolveAdjoint, EnzymeVJP,
                            ForwardDiffOverAdjoint, ForwardDiffSensitivity, ForwardLSS,
                            ForwardSensitivity, GaussAdjoint, InterpolatingAdjoint, NILSAS,
                            NILSS, QuadratureAdjoint, ReverseDiffAdjoint, ReverseDiffVJP,
                            SteadyStateAdjoint, TrackerAdjoint, TrackerVJP, ZygoteAdjoint,
                            ZygoteVJP
    using Tracker: Tracker
    using Zygote: Zygote
end

const CRC = ChainRulesCore

@reexport using ADTypes, Lux

include("ffjord.jl")
include("neural_de.jl")
include("spline_layer.jl")
include("tensor_product.jl")
include("collocation.jl")
include("hnn.jl")
include("multiple_shooting.jl")

export NeuralODE, NeuralDSDE, NeuralSDE, NeuralCDDE, NeuralDAE, AugmentedNDELayer,
       NeuralODEMM, TensorLayer, SplineLayer
export NeuralHamiltonianDE, HamiltonianNN
export FFJORD, FFJORDDistribution
export TensorProductBasisFunction, ChebyshevBasis, SinBasis, CosBasis, FourierBasis,
       LegendreBasis, PolynomialBasis
export DimMover

export EpanechnikovKernel, UniformKernel, TriangularKernel, QuarticKernel, TriweightKernel,
       TricubeKernel, GaussianKernel, CosineKernel, LogisticKernel, SigmoidKernel,
       SilvermanKernel
export collocate_data

export multiple_shoot

# Reexporting only certain functions from SciMLSensitivity
export BacksolveAdjoint, QuadratureAdjoint, GaussAdjoint, InterpolatingAdjoint,
       TrackerAdjoint, ZygoteAdjoint, ReverseDiffAdjoint, ForwardSensitivity,
       ForwardDiffSensitivity, ForwardDiffOverAdjoint, SteadyStateAdjoint, ForwardLSS,
       AdjointLSS, NILSS, NILSAS
export TrackerVJP, ZygoteVJP, EnzymeVJP, ReverseDiffVJP

end
