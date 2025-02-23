@testitem "Tests for OTFlow Layer Functionality" begin
    using Lux, LuxCore, Random, LinearAlgebra, Test, ComponentArrays, Flux, DiffEqFlux
    rng = Xoshiro(0)
    d = 2 
    m = 4 
    r = 2 
    otflow = OTFlow(d, m; r=r)
    ps, st = Lux.setup(rng, otflow)
    ps = ComponentArray(ps)

    x = Float32[1.0, 2.0]
    t = 0.5f0

    @testitem "Forward Pass" begin
        (v, tr), st_new = otflow((x, t), ps, st)
        @test length(v) == d 
        @test isa(tr, Float32)
        @test st_new == st 
    end

    @testitem "Potential Function" begin
        phi = potential(x, t, ps)
        @test isa(phi, Float32)
    end

    @testitem "Gradient Consistency" begin
        grad = gradient(x, t, ps, d)
        (v, _), _ = otflow((x, t), ps, st)
        @test length(grad) == d
        @test grad ≈ -v atol=1e-5  # v = -∇Φ
    end

    @testitem "Trace Consistency" begin
        tr_manual = trace(x, t, ps, d)
        (_, tr_forward), _ = otflow((x, t), ps, st)
        @test tr_manual ≈ -tr_forward atol=1e-5
    end

    @testitem "ODE Integration" begin
        x0 = Float32[1.0, 1.0]
        tspan = (0.0f0, 1.0f0)
        x_traj, t_vec = simple_ode_solve(otflow, x0, tspan, ps, st; dt=0.01f0)
        @test size(x_traj) == (d, length(t_vec))
        @test all(isfinite, x_traj)
        @test x_traj[:, end] != x0
    end

    @testitem "Loss Function" begin
        loss_val = simple_loss(x, t, otflow, ps)
        @test isa(loss_val, Float32)
        @test isfinite(loss_val)
    end

    @testitem "Manual Gradient" begin
        grads = manual_gradient(x, t, otflow, ps)
        @test haskey(grads, :w) && length(grads.w) == m
        @test haskey(grads, :A) && size(grads.A) == (r, d+1)
        @test haskey(grads, :b) && length(grads.b) == d+1
        @test haskey(grads, :c) && isa(grads.c, Float32)
        @test haskey(grads, :K0) && size(grads.K0) == (m, d+1)
        @test haskey(grads, :K1) && size(grads.K1) == (m, m)
        @test haskey(grads, :b0) && length(grads.b0) == m
        @test haskey(grads, :b1) && length(grads.b1) == m
    end
end