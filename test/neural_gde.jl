using DiffEqFlux, ComponentArrays, GeometricFlux, GraphSignals, OrdinaryDiffEq, Random,
    Test, OptimizationOptimisers, Optimization, Statistics
import Flux

# Fully Connected Graph
adj_mat = FeaturedGraph(Float32[0 1 1 1
    1 0 1 1
    1 1 0 1
    1 1 1 0])

features = [-10.0f0 -9.0f0 9.0f0 10.0f0
    0.0f0 0.0f0 0.0f0 0.0f0]

target = Float32[1.0 1.0 0.0 0.0
    0.0 0.0 1.0 1.0]

model = Chain(NeuralODE(WithGraph(adj_mat, GCNConv(2 => 2)), (0.0f0, 1.0f0), Tsit5();
        save_everystep = false, reltol = 1e-3, abstol = 1e-3, save_start = false),
    x -> reshape(Array(x), size(x)[1:2]))

ps, st = Lux.setup(Xoshiro(0), model)
ps = ComponentArray(ps)

logitcrossentropy(ŷ, y; dims = 1) = mean(.-sum(y .* logsoftmax(ŷ; dims); dims))

lux_model = Lux.Experimental.StatefulLuxLayer(model, ps, st)

initial_loss = logitcrossentropy(lux_model(features, ps), target)

loss_function(p) = logitcrossentropy(lux_model(features, p), target)

function callback(p, l)
    @info "[NeuralGraphODE] Loss: $l"
    return false
end

optfunc = Optimization.OptimizationFunction((x, p) -> loss_function(x),
    Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optfunc, ps)
res = Optimization.solve(optprob, Adam(0.1); callback, maxiters = 100)

updated_loss = logitcrossentropy(lux_model(features, ps), target)

@test_broken updated_loss < initial_loss
