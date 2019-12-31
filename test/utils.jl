using Flux, Tracker, Test
using DiffEqFlux: destructure, restructure

model = Chain(Dense(10, 5, relu), Dense(5, 2))

p = destructure(model)

m2 = restructure(model, p)

x = rand(10)

@test model(x) == m2(x)

sz = ((2,50),(50,),(2,),(50,2))
xs = rand(sum(prod.(sz)))
out = restructure(sz,xs)
@test all(size(out[i]) == sz[i] for i in 1:4)
