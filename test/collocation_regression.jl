using DiffEqFlux, OrdinaryDiffEq, Test

function f(u, p, t)
    p .* u
end
rc = 62
ps = repeat([-0.001], rc)
tspan = (7.0, 84.0)
u0 = 3.4 .+ ones(rc)
t = collect(range(minimum(tspan), stop = maximum(tspan), length = 157))
prob = ODEProblem(f, u0, tspan, ps)
data = Array(solve(prob, Tsit5(), saveat = t, abstol = 1e-12, reltol = 1e-12))
ptest = ones(rc)

u′, u = collocate_data(data, t, SigmoidKernel())
@test sum(abs2, u - data) < 1e-8

function loss(p)
    cost = zero(first(p))
    for i = 1:size(u′, 2)
        _du = f(@view(u[:, i]), p, t[i])
        u′i = @view u′[:, i]
        cost += sum(abs2, u′i .- _du)
    end
    sqrt(cost)
end
@test loss(ptest) ≈ 418.3400017500223
