using SciMLTesting, DiffEqFlux, Test

run_qa(
    DiffEqFlux;
    explicit_imports = true,
    # `ambiguities = false` in test_all + a separate non-recursive ambiguity check
    # historically; keep ambiguities on but non-recursive (recursive hits the deep
    # Lux/SciMLSensitivity stack and is not DiffEqFlux's responsibility).
    aqua_kwargs = (; ambiguities = (; recursive = false)),
    api_docs = false,
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

@testset "DiffEqFlux-owned public API docs" begin
    public_api = [
        :NeuralDELayer,
        :NeuralSDELayer,
        :CNFLayer,
        :NeuralODE,
        :NeuralDSDE,
        :NeuralSDE,
        :NeuralCDDE,
        :NeuralDAE,
        :AugmentedNDELayer,
        :NeuralODEMM,
        :FFJORD,
        :FFJORDDistribution,
        :DimMover,
        :CollocationKernel,
        :EpanechnikovKernel,
        :UniformKernel,
        :TriangularKernel,
        :QuarticKernel,
        :TriweightKernel,
        :TricubeKernel,
        :GaussianKernel,
        :CosineKernel,
        :LogisticKernel,
        :SigmoidKernel,
        :SilvermanKernel,
        :collocate_data,
        :multiple_shoot,
        :group_ranges,
    ]

    @testset "source docstrings" begin
        for name in public_api
            @test Base.Docs.doc(Base.Docs.Binding(DiffEqFlux, name)) !== nothing
        end
    end

    docs_entries = Set{String}()
    docs_root = normpath(joinpath(@__DIR__, "..", "..", "docs", "src"))
    for (root, _, files) in walkdir(docs_root)
        for file in files
            endswith(file, ".md") || continue
            path = joinpath(root, file)
            in_docs_block = false
            for line in eachline(path)
                stripped = strip(line)
                if stripped == "```@docs"
                    in_docs_block = true
                elseif stripped == "```"
                    in_docs_block = false
                elseif in_docs_block && !isempty(stripped)
                    push!(docs_entries, stripped)
                end
            end
        end
    end

    rendered_names = Dict(
        :NeuralDELayer => "DiffEqFlux.NeuralDELayer",
        :NeuralSDELayer => "DiffEqFlux.NeuralSDELayer",
        :CNFLayer => "DiffEqFlux.CNFLayer",
        :CollocationKernel => "DiffEqFlux.CollocationKernel",
        :group_ranges => "DiffEqFlux.group_ranges",
    )

    @testset "rendered @docs entries" begin
        for name in public_api
            rendered_name = get(rendered_names, name, string(name))
            @test rendered_name in docs_entries
        end
    end
end
