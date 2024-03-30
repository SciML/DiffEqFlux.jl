module DiffEqFlux

import PrecompileTools

PrecompileTools.@recompile_invalidations begin
    using ADTypes, ChainRulesCore, ComponentArrays, ConcreteStructs, Functors,
          LinearAlgebra, Lux, LuxCore, Random, Reexport, SciMLBase, SciMLSensitivity

    # AD Packages
    using ForwardDiff, Tracker, Zygote

    # FFJORD Specific
    using Distributions, DistributionsAD
end

import ChainRulesCore as CRC
import LuxCore: AbstractExplicitLayer, AbstractExplicitContainerLayer
import Lux.Experimental: StatefulLuxLayer

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
export TensorProductBasisFunction,
       ChebyshevBasis, SinBasis, CosBasis, FourierBasis, LegendreBasis, PolynomialBasis
export DimMover

export EpanechnikovKernel, UniformKernel, TriangularKernel, QuarticKernel, TriweightKernel,
       TricubeKernel, GaussianKernel, CosineKernel, LogisticKernel, SigmoidKernel,
       SilvermanKernel
export collocate_data

export multiple_shoot

# Reexporting only certain functions from SciMLSensitivity
export BacksolveAdjoint, QuadratureAdjoint, GaussAdjoint, InterpolatingAdjoint,
       TrackerAdjoint, ZygoteAdjoint, ReverseDiffAdjoint, ForwardSensitivity,
       ForwardDiffSensitivity, ForwardDiffOverAdjoint, SteadyStateAdjoint,
       ForwardLSS, AdjointLSS, NILSS, NILSAS
export TrackerVJP, ZygoteVJP, EnzymeVJP, ReverseDiffVJP

end
