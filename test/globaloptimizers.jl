### Global optimizer tests

# TikTak

f(x) = sum(x -> abs2(x - 1), x), nothing
res = DiffEqFlux.sciml_train(f, rand(100), TikTak(10), 
                       local_method = NLopt.LN_BOBYQA,
                       lower_bounds = -10*ones(100), 
                       upper_bounds = 10*ones(100))

@test isapprox(res.minimizer, ones(100), rtol = 1e-10)
