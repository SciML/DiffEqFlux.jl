module DiffEqFlux

using ADTypes: ADTypes, AutoForwardDiff, AutoZygote
using Boltz: Boltz, Basis, Layers
using ChainRulesCore: ChainRulesCore
using ConcreteStructs: @concrete
using Distributions: Distributions, ContinuousMultivariateDistribution, Distribution, logpdf
using LinearAlgebra: LinearAlgebra, Diagonal, det, tr, mul!
using Lux: Lux, Chain, Dense, StatefulLuxLayer, FromFluxAdaptor
using LuxCore: LuxCore, AbstractLuxLayer, AbstractLuxContainerLayer, AbstractLuxWrapperLayer
using LuxLib: LuxLib, batched_matmul
using Random: Random, AbstractRNG, randn!
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
using Static: True, False

const CRC = ChainRulesCore

# The neural-network construction surface that DiffEqFlux reexports (see the `export`
# blocks at the bottom of this file), so that `using DiffEqFlux` on its own is enough to
# build the network a `NeuralODE`/`NeuralDSDE`/`FFJORD` wraps and to pick an AD backend
# for it. Every name stays owned and documented upstream:
#
#   * layers and the Flux adaptor from Lux,
#   * the layer contract (`setup`, `apply`, ...) from LuxCore,
#   * the `Auto*` differentiation selectors from ADTypes,
#   * the `Layers`/`Basis` model zoo modules from Boltz.
using Lux: Conv, FlattenLayer, GroupNorm, MaxPool, MeanPool, Training,
    WrappedFunction, f32, f64
using LuxCore: apply, initialparameters, initialstates, parameterlength, setup,
    statelength, testmode, trainmode
using ADTypes: AbstractADType, AutoChainRules, AutoDiffractor, AutoEnzyme,
    AutoFastDifferentiation, AutoFiniteDiff, AutoFiniteDifferences, AutoGTPSA,
    AutoHyperHessians, AutoModelingToolkit, AutoMooncake, AutoMooncakeForward,
    AutoPolyesterForwardDiff, AutoReactant, AutoReverseDiff, AutoSparse, AutoSymbolics,
    AutoTaylorDiff, AutoTracker

fixed_state_type(_) = true
fixed_state_type(::Layers.HamiltonianNN{True}) = true
fixed_state_type(::Layers.HamiltonianNN{False}) = false

include("ffjord.jl")
include("neural_de.jl")

include("collocation.jl")
include("multiple_shooting.jl")

export NeuralODE, NeuralDSDE, NeuralSDE, NeuralCDDE, NeuralDAE, AugmentedNDELayer,
    NeuralODEMM
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

# Reexported neural-network construction surface; approved via `reexports_allow` in
# test/QA/qa_tests.jl and documented in docs/src/reexports.md.
export Lux, Chain, Dense, Conv, MaxPool, MeanPool, FlattenLayer, GroupNorm,
    WrappedFunction, StatefulLuxLayer, FromFluxAdaptor, Training, f32, f64
export LuxCore, AbstractLuxLayer, AbstractLuxContainerLayer, AbstractLuxWrapperLayer,
    setup, apply, initialparameters, initialstates, parameterlength, statelength,
    testmode, trainmode
export ADTypes, AbstractADType, AutoChainRules, AutoDiffractor, AutoEnzyme,
    AutoFastDifferentiation, AutoFiniteDiff, AutoFiniteDifferences, AutoForwardDiff,
    AutoGTPSA, AutoHyperHessians, AutoModelingToolkit, AutoMooncake, AutoMooncakeForward,
    AutoPolyesterForwardDiff, AutoReactant, AutoReverseDiff, AutoSparse, AutoSymbolics,
    AutoTaylorDiff, AutoTracker, AutoZygote
export Boltz, Basis, Layers

# Precompilation workload - must be at the end
include("precompilation.jl")

end
