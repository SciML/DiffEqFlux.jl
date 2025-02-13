@testitem "OTFlow Tests" begin
    using DiffEqFlux
    using Lux
    using Random
    using Distributions
    using Test

    rng = Random.default_rng()
    
    # Test basic constructor and initialization
    @testset "Constructor" begin
        model = Dense(2 => 1)
        tspan = (0.0, 1.0)
        input_dims = (2,)
        
        flow = OTFlow(model, tspan, input_dims)
        @test flow isa OTFlow
        @test flow.tspan == tspan
        @test flow.input_dims == input_dims
        
        # Test with base distribution
        base_dist = MvNormal(2, 1.0)
        flow_with_dist = OTFlow(model, tspan, input_dims; basedist=base_dist)
        @test flow_with_dist.basedist == base_dist
    end

    @testset "Forward Pass" begin
        model = Dense(2 => 1)
        tspan = (0.0, 1.0)
        input_dims = (2,)
        flow = OTFlow(model, tspan, input_dims)
        
        ps, st = Lux.setup(rng, flow)
        x = randn(rng, 2, 10) # 10 samples of 2D data
        
        # Test forward pass
        output, new_st = flow(x, ps, st)
        @test size(output) == (1, 10) # log probabilities
        @test new_st isa NamedTuple
    end

    @testset "Distribution Interface" begin
        model = Dense(2 => 1)
        tspan = (0.0, 1.0)
        input_dims = (2,)
        flow = OTFlow(model, tspan, input_dims)
        
        ps, st = Lux.setup(rng, flow)
        dist = OTFlowDistribution(flow, ps, st)
        
        # Test sampling
        x = rand(dist, 5)
        @test size(x) == (2, 5)
        
        # Test log pdf
        logp = logpdf(dist, x[:, 1])
        @test logp isa Real
    end

    @testset "Base Distribution" begin
        model = Dense(2 => 1)
        tspan = (0.0, 1.0)
        input_dims = (2,)
        base_dist = MvNormal(zeros(2), I)
        
        flow = OTFlow(model, tspan, input_dims; basedist=base_dist)
        ps, st = Lux.setup(rng, flow)
        
        x = randn(rng, 2, 5)
        output, new_st = flow(x, ps, st)
        @test size(output) == (1, 5)
    end
end