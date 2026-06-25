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
        # Non-public names accessed by qualifier. Each is another package's internal
        # API (extension points / solver-internal abstract types) that DiffEqFlux
        # uses or extends; they become public as the base libs declare them.
        all_qualified_accesses_are_public = (;
            ignore = (
                :AbstractODEAlgorithm,    # SciMLBase
                :BasicEnsembleAlgorithm,  # SciMLBase
                :successful_retcode,      # SciMLBase
                :Fix2,                    # Base
                :HamiltonianNN,           # Boltz.Layers
                :_logpdf,                 # Distributions (extension point)
                :_rand!,                  # Distributions (extension point)
                :initialstates,           # LuxCore (extension point)
                :setup,                   # LuxCore (extension point)
                :recursive_eltype,        # Lux
            ),
        ),
    ),
)
