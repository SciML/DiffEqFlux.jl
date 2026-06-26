using SciMLTesting, DiffEqFlux, Test

run_qa(
    DiffEqFlux;
    explicit_imports = true,
    # `ambiguities = false` in test_all + a separate non-recursive ambiguity check
    # historically; keep ambiguities on but non-recursive (recursive hits the deep
    # Lux/SciMLSensitivity stack and is not DiffEqFlux's responsibility).
    aqua_kwargs = (; ambiguities = (; recursive = false)),
    ei_kwargs = (
        # Reexported module names: DiffEqFlux deliberately
        # `@reexport using ADTypes, Lux, Boltz`, so `Boltz`/`Layers` (Boltz) and
        # `LuxLib` (Lux) are pulled in via those reexports, not bare implicit imports.
        no_implicit_imports = (; skip = (Base, Core, ADTypes, Lux, Boltz)),
        # FFJORDDistribution overrides Distributions' internal `_logpdf`/`_rand!`
        # extension points; these remain non-public in Distributions (0.25.126).
        all_qualified_accesses_are_public = (;
            ignore = (
                :_logpdf,  # Distributions (extension point)
                :_rand!,   # Distributions (extension point)
            ),
        ),
    ),
)
