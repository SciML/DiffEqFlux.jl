using Flux, DiffEqFlux, Test, StaticArrays

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
x = rand(2)

@test fs(x,pd) ≈ fd(x,pd)
fdgrad = Flux.Zygote.gradient((x,p)->sum(fd(x,p)),x,pd)
fsgrad = Flux.Zygote.gradient((x,p)->sum(fs(x,p)),x,pd)
@test fdgrad[1] ≈ fsgrad[1]
@test fdgrad[2] ≈ fsgrad[2] rtol=1e-5

# Now test vs Zygote
struct TestDense{F,F2} <: DiffEqFlux.FastLayer
  out::Int
  in::Int
  σ::F
  initial_params::F2
  function TestDense(in::Integer, out::Integer, σ = identity;
                 initW = Flux.glorot_uniform, initb = Flux.zeros)
    initial_params() = vcat(vec(initW(out, in)),initb(out))
    new{typeof(σ),typeof(initial_params)}(out,in,σ,initial_params)
  end
end
(f::TestDense)(x,p) = f.σ.(reshape(p[1:(f.out*f.in)],f.out,f.in)*x .+ p[(f.out*f.in+1):end])
ts = StaticDense(2,25,tanh)
testgrad = Flux.Zygote.gradient((x,p)->sum(ts(x,p)),x,pd)
@test fdgrad[1] ≈ testgrad[1]
@test fdgrad[2] ≈ testgrad[2] rtol=1e-5

fsgrad = Flux.Zygote.gradient((x,p)->sum(fs(x,p)),SVector{2}(x),SVector{75}(pd))
@test fsgrad[1] isa SArray
@test fsgrad[2] isa SArray
