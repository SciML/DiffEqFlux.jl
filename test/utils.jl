using Flux, Tracker, Test
using DiffEqFlux: destructure, restructure

model = Chain(Dense(10, 5, relu), Dense(5, 2))

p = destructure(model)

m2 = restructure(model, p)

@test !Tracker.isleaf(m2[1].W)

x = rand(10)

@test model(x) == m2(x)
