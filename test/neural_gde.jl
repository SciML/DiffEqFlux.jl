using DiffEqFlux, GeometricFlux, GraphSignals, OrdinaryDiffEq, Test

# Fully Connected Graph
adj_mat = FeaturedGraph(Float32[0 1 1 1
                                1 0 1 1
                                1 1 0 1
                                1 1 1 0])

features = [-10.0f0 -9.0f0 9.0f0 10.0f0
            0.0f0 0.0f0 0.0f0 0.0f0]

target = [1.0 1.0 0.0 0.0
          0.0 0.0 1.0 1.0]

model = Flux.Chain(NeuralODE(GCNConv(adj_mat, 2 => 2),
                             (0.0f0, 1.0f0), Tsit5(), save_everystep = false,
                             reltol = 1e-3, abstol = 1e-3, save_start = false),
                   x -> reshape(cpu(x), size(x)[1:2]))

ps = Flux.params(model)
opt = ADAM(0.1)

initial_loss = Flux.Losses.logitcrossentropy(model(features), target)

# for i in 1:100
#     gs = gradient(() -> Flux.Losses.logitcrossentropy(model(features), target), ps)
#     Flux.Optimise.update!(opt, ps, gs)
# end
updated_loss = Flux.Losses.logitcrossentropy(model(features), target)

@test_broken updated_loss < initial_loss
