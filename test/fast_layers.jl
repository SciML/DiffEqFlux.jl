using Flux, DiffEqFlux

f,p = FastDense(2,25,tanh)
length(f)
f(ones(2),p)

f1,p1 = FastDense(2,25,tanh)
f2,p2 = FastDense(25,2,tanh)
FastChain(f1,f2)[1](ones(2),[p1;p2]) == f2(f1(ones(2),p1),p2)

f,p = FastChain(FastDense(2,25,tanh),FastDense(25,2,tanh))
f(ones(2),p) == f2(f1(ones(2),p[1:length(p1)]),p[length(p1)+1:end])
