using Flux, DiffEqFlux, Test

fd = FastDense(2,25,tanh)
pd = initial_params(fd)
fd(ones(2),pd)

f1 = FastDense(2,25,tanh)
f2 = FastDense(25,2,tanh)
p1 = initial_params(f1)
p2 = initial_params(f2)
@test FastChain(f1,f2)(ones(2),[p1;p2]) == f2(f1(ones(2),p1),p2)

f = FastChain(FastDense(2,25,tanh),FastDense(25,2,tanh))
p = initial_params(f)
@test f(ones(2),p) == f2(f1(ones(2),p[1:length(p1)]),p[length(p1)+1:end])

fs = StaticDense(2,25,tanh)
@test fs(ones(2),pd) ≈ fd(ones(2),pd)
