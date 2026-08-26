using SciMLTesting, DiffEqFlux, Test

# The dependency-owned names DiffEqFlux deliberately reexports, so that
# `using DiffEqFlux` is enough to build a neural differential equation end to end:
# the SciMLSensitivity adjoint/VJP choices, the Lux layers and LuxCore layer contract
# used to build the network, the ADTypes `Auto*` selectors, and the Boltz `Layers`/
# `Basis` model zoo modules. Each name is owned and documented upstream; kept in sync
# with the `export` blocks in src/DiffEqFlux.jl and the list in docs/src/reexports.md.
const REEXPORTS = (
    # SciMLSensitivity: the solver-facing sensitivity facade.
    :AdjointLSS, :BacksolveAdjoint, :EnzymeVJP, :ForwardDiffOverAdjoint,
    :ForwardDiffSensitivity, :ForwardLSS, :ForwardSensitivity, :GaussAdjoint,
    :InterpolatingAdjoint, :NILSAS, :NILSS, :QuadratureAdjoint,
    :ReverseDiffAdjoint, :ReverseDiffVJP, :SteadyStateAdjoint, :TrackerAdjoint,
    :TrackerVJP, :ZygoteAdjoint, :ZygoteVJP,
    # Lux: network construction.
    :Chain, :Conv, :Dense, :FlattenLayer, :FromFluxAdaptor, :GroupNorm, :Lux, :MaxPool,
    :MeanPool, :SamePad, :StatefulLuxLayer, :Training, :WrappedFunction, :f32, :f64,
    # LuxCore: the layer contract.
    :AbstractLuxContainerLayer, :AbstractLuxLayer, :AbstractLuxWrapperLayer, :LuxCore,
    :apply, :initialparameters, :initialstates, :parameterlength, :setup, :statelength,
    :testmode, :trainmode,
    # ADTypes: the differentiation backend selectors.
    :ADTypes, :AbstractADType, :AutoChainRules, :AutoDiffractor, :AutoEnzyme,
    :AutoFastDifferentiation, :AutoFiniteDiff, :AutoFiniteDifferences, :AutoForwardDiff,
    :AutoGTPSA, :AutoHyperHessians, :AutoModelingToolkit, :AutoMooncake,
    :AutoMooncakeForward, :AutoPolyesterForwardDiff, :AutoReactant, :AutoReverseDiff,
    :AutoSparse, :AutoSymbolics, :AutoTaylorDiff, :AutoTracker, :AutoZygote,
    # Boltz: the model zoo modules the examples build layers from.
    :Basis, :Boltz, :Layers,
)

run_qa(
    DiffEqFlux;
    reexports_allow = REEXPORTS,
    # `ambiguities = false` in test_all + a separate non-recursive ambiguity check
    # historically; keep ambiguities on but non-recursive (recursive hits the deep
    # Lux/SciMLSensitivity stack and is not DiffEqFlux's responsibility).
    aqua_kwargs = (; ambiguities = (; recursive = false)),
    ei_kwargs = (
        # `FFJORDDistribution` implements Distributions' documented extension points
        # `_logpdf`/`_rand!` (a custom `ContinuousMultivariateDistribution` must define
        # these; see `Distributions.common`: "Instead of `logpdf` one should implement
        # `_logpdf(d, x)`). They are deliberately underscore-prefixed and not public,
        # so the access can be neither migrated to a public owner nor made public.
        all_qualified_accesses_are_public = (;
            ignore = (
                :_logpdf,
                :_rand!,
            ),
        ),
    ),
)

@testset "Reexport surface" begin
    # Every approved reexport must actually be reachable from `using DiffEqFlux`, so the
    # allow-list cannot drift into approving names the package no longer provides.
    # `isdefined(@__MODULE__, ...)` tests the property directly: this file's
    # `using DiffEqFlux` is what has to bring the name into scope.
    @testset "$name" for name in REEXPORTS
        @test name in names(DiffEqFlux)
        @test isdefined(@__MODULE__, name)
    end
end
