using Flux, DiffEqFlux

f = FastDense(2,25,tanh)
f(ones(2),initial_params(f))

f1 = FastDense(2,25,tanh)
f2 = FastDense(25,2,tanh)
p1 = initial_params(f1)
p2 = initial_params(f2)
FastChain(f1,f2)[1](ones(2),[p1;p2]) == f2(f1(ones(2),p1),p2)

f = FastChain(FastDense(2,25,tanh),FastDense(25,2,tanh))
p = initial_params(f)
f(ones(2),p) == f2(f1(ones(2),p[1:length(p1)]),p[length(p1)+1:end])

f = StaticDense(2,25,tanh)
f(ones(2),initial_params(f))
