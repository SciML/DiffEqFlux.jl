using OrdinaryDiffEq, DiffEqSensitivity, Flux, Test
pa = [1.0]
u0 = [3.0]
function model1()
  prob = ODEProblem((u, p, t) -> 1.01u .* p, u0, (0.0, 1.0), pa)

  function prob_func(prob, i, repeat)
    remake(prob, u0 = 0.5 .+ i/100 .* prob.u0)
  end

  ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
  sim = solve(ensemble_prob, Tsit5(), EnsembleSerial(), saveat = 0.1, trajectories = 100)
end

# loss function
loss() = sum(abs2,1.0.-Array(model1()))

data = Iterators.repeated((), 10)

cb = function () # callback function to observe training
  @show loss()
end

opt = ADAM(0.1)
println("Starting to train")
l1 = loss()
Flux.@epochs 10 Flux.train!(loss, params([pa,u0]), data, opt; cb = cb)
l2 = loss()
@test 10l2 < l1

function model2()
  prob = ODEProblem((u, p, t) -> 1.01u .* p, u0, (0.0, 1.0), pa)

  function prob_func(prob, i, repeat)
    remake(prob, u0 = 0.5 .+ i/100 .* prob.u0)
  end

  ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
  sim = solve(ensemble_prob, Tsit5(), EnsembleSerial(), saveat = 0.1, trajectories = 100).u
end
loss() = sum(abs2,[sum(abs2,1.0.-u) for u in model2()])

pa = [1.0]
u0 = [3.0]
opt = ADAM(0.1)
println("Starting to train")
l1 = loss()
Flux.@epochs 10 Flux.train!(loss, params([pa,u0]), data, opt; cb = cb)
l2 = loss()
@test 10l2 < l1

function model3()
  prob = ODEProblem((u, p, t) -> 1.01u .* p, u0, (0.0, 1.0), pa)

  function prob_func(prob, i, repeat)
    remake(prob, u0 = 0.5 .+ i/100 .* prob.u0)
  end

  ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
  sim = solve(ensemble_prob, Tsit5(), EnsembleThreads(), saveat = 0.1, trajectories = 100)
end

# loss function
loss() = sum(abs2,1.0.-Array(model3()))

data = Iterators.repeated((), 10)

cb = function () # callback function to observe training
  @show loss()
end

pa = [1.0]
u0 = [3.0]
opt = ADAM(0.1)
println("Starting to train")
l1 = loss()
Flux.@epochs 10 Flux.train!(loss, params([pa,u0]), data, opt; cb = cb)
l2 = loss()
@test 10l2 < l1

function model4()
  prob = ODEProblem((u, p, t) -> 1.01u .* p, u0, (0.0, 1.0), pa)

  function prob_func(prob, i, repeat)
    remake(prob, u0 = 0.5 .+ i/100 .* prob.u0)
  end

  ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
  sim = solve(ensemble_prob, Tsit5(), EnsembleDistributed(), saveat = 0.1, trajectories = 100)
end

# loss function
loss() = sum(abs2,1.0.-Array(model4()))

data = Iterators.repeated((), 10)

cb = function () # callback function to observe training
  @show loss()
end

pa = [1.0]
u0 = [3.0]
opt = ADAM(0.1)
println("Starting to train")
l1 = loss()
Flux.@epochs 10 Flux.train!(loss, Flux.params([pa,u0]), data, opt; cb = cb)
l2 = loss()
@test 10l2 < l1
