# One point test
using Flux, ReverseDiff, LagrangianNN

m, k, b = 1, 1, 1

X = rand(2,1)
Y = -k.*X[1]/m

g = Chain(Dense(2, 10, σ), Dense(10,1))
model = LagrangianNN(g)
params = model.params
re = model.re

# some toy loss function
function loss(x, y, p)
    nn = x -> model(x,p)
    out = sum((y .- (nn(x))).^2)
    out
end
opt = ADAM(0.01)
epochs = 100

for epoch in 1:epochs
    x, y = X, Y
    gs = ReverseDiff.gradient(p -> loss(x, y, p), params)
    Flux.Optimise.update!(opt, params, gs)
    @show loss(x,y,params)
end
