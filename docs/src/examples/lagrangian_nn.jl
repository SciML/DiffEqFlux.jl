# One point test
using Flux, ReverseDiff, LagrangianNN

m, k, b = 1, 1, 1

X = rand(2,1)
Y = -k.*X[1]/m

g = Chain(Dense(2, 10, σ), Dense(10,1))
model = LagrangianNN(g)
params = model.params
re = model.re

loss(x, y, p) = sum(abs2, y .- model(x, p))

opt = ADAM(0.01)
epochs = 100

for epoch in 1:epochs
    x, y = X, Y
    gs = ReverseDiff.gradient(p -> loss(x, y, p), params)
    Flux.Optimise.update!(opt, params, gs)
    @show loss(x,y,params)
end
