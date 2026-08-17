using SciMLTesting, DiffEqFlux

run_qa(
    DiffEqFlux;
    # These explicitly exported SciMLSensitivity adjoint algorithms are part of
    # DiffEqFlux's documented solver-facing facade.
    reexports_allow = (
        :AdjointLSS, :BacksolveAdjoint, :EnzymeVJP, :ForwardDiffOverAdjoint,
        :ForwardDiffSensitivity, :ForwardLSS, :ForwardSensitivity, :GaussAdjoint,
        :InterpolatingAdjoint, :NILSAS, :NILSS, :QuadratureAdjoint,
        :ReverseDiffAdjoint, :ReverseDiffVJP, :SteadyStateAdjoint, :TrackerAdjoint,
        :TrackerVJP, :ZygoteAdjoint, :ZygoteVJP,
    ),
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
