using SciMLTesting, DiffEqFlux

const REEXPORTED_API = Tuple(
    n for n in names(DiffEqFlux; all = false, imported = false) if
        n !== nameof(DiffEqFlux) &&
        isdefined(DiffEqFlux, n) &&
        parentmodule(getproperty(DiffEqFlux, n)) !== DiffEqFlux
)

run_qa(
    DiffEqFlux;
    explicit_imports = true,
    # `ambiguities = false` in test_all + a separate non-recursive ambiguity check
    # historically; keep ambiguities on but non-recursive (recursive hits the deep
    # Lux/SciMLSensitivity stack and is not DiffEqFlux's responsibility).
    aqua_kwargs = (; ambiguities = (; recursive = false)),
    api_docs_kwargs = (;
        rendered = true,
        ignore = REEXPORTED_API,
        rendered_ignore = REEXPORTED_API,
    ),
    ei_kwargs = (
        # `FFJORDDistribution` implements Distributions' documented extension points
        # `_logpdf`/`_rand!` (a custom `ContinuousMultivariateDistribution` must define
        # these; see `Distributions.common`: "Instead of `logpdf` one should implement
        # `_logpdf(d, x)`"). They are deliberately underscore-prefixed and not public,
        # so the access can be neither migrated to a public owner nor made public.
        all_qualified_accesses_are_public = (;
            ignore = (
                :_logpdf,  # Distributions extension point (non-public by convention)
                :_rand!,   # Distributions extension point (non-public by convention)
            ),
        ),
    ),
)
