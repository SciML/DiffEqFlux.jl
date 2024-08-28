module DiffEqFlux

using ADTypes: ADTypes, AutoForwardDiff, AutoZygote
using ChainRulesCore: ChainRulesCore
using ConcreteStructs: @concrete
using DataInterpolations: DataInterpolations
using Distributions: Distributions, ContinuousMultivariateDistribution, Distribution, logpdf
using DistributionsAD: DistributionsAD
using ForwardDiff: ForwardDiff
using LinearAlgebra: LinearAlgebra, Diagonal, det, tr, mul!
using Lux: Lux, Chain, Dense, StatefulLuxLayer, FromFluxAdaptor
using LuxCore: LuxCore, AbstractExplicitLayer, AbstractExplicitContainerLayer
using LuxLib: batched_matmul
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
using Setfield: @set!
using Zygote: Zygote

const CRC = ChainRulesCore

@reexport using ADTypes, Lux, Boltz

include("ffjord.jl")
include("neural_de.jl")
include("spline_layer.jl")
include("collocation.jl")
include("hnn.jl")
include("multiple_shooting.jl")
include("deprecated.jl")

export NeuralODE, NeuralDSDE, NeuralSDE, NeuralCDDE, NeuralDAE, AugmentedNDELayer,
       NeuralODEMM, SplineLayer
export NeuralHamiltonianDE, HamiltonianNN
export FFJORD, FFJORDDistribution
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
