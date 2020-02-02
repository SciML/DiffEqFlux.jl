paramlength(f) = 0
initial_params(f) = Float32[]
initial_params(f::Chain) = Flux.destructure(f)[1]

struct FastChain{T<:Tuple}
  layers::T
  function FastChain(xs...)
    layers = getfunc.(xs)
    new{typeof(layers)}(layers)
  end
end
getfunc(x) = x
getfunc(x::Tuple) = first(x)
getparams(x) = Float32[]
getparams(x::Tuple) = last(x)

applychain(::Tuple{}, x, p) = x
applychain(fs::Tuple, x, p) = applychain(Base.tail(fs), first(fs)(x,p[1:paramlength(first(fs))]), p[(paramlength(first(fs))+1):end])
(c::FastChain)(x,p) = applychain(c.layers, x, p)
initial_params(c::FastChain) = vcat(initial_params.(c.layers)...)

struct FastDense{F,F2} <: Function
  out::Int
  in::Int
  σ::F
  initial_params::F2
  function FastDense(in::Integer, out::Integer, σ = identity;
                 initW = Flux.glorot_uniform, initb = Flux.zeros)
    initial_params() = vcat(vec(initW(out, in)),initb(out))
    new{typeof(σ),typeof(initial_params)}(out,in,σ,initial_params)
  end
end
# (f::FastDense)(x,p) = f.σ.(reshape(uview(p,1:(f.out*f.in)),f.out,f.in)*x .+ uview(p,(f.out*f.in+1):lastindex(p)))
(f::FastDense)(x,p) = f.σ.(reshape(p[1:(f.out*f.in)],f.out,f.in)*x .+ p[(f.out*f.in+1):end])
paramlength(f::FastDense) = f.out*(f.in + 1)
initial_params(f::FastDense) = f.initial_params()

struct StaticDense{out,in,F,F2} <: Function
  σ::F
  initial_params::F2
  function StaticDense(in::Integer, out::Integer, σ = identity;
                 initW = Flux.glorot_uniform, initb = Flux.zeros)
    initial_params() = vcat(vec(initW(out, in)),initb(out))
    new{out,in,typeof(σ),typeof(initial_params)}(σ,initial_params)
  end
end
(f::StaticDense{out,in})(x,p) where {out,in} = f.σ.(SMatrix{out,in}(uview(p,1:(out*in)))*x .+ SVector{out}(uview(p,(out*in+1):lastindex(p))))
ZygoteRules.@adjoint function (f::StaticDense{out,in})(x,p) where {out,in}
  W = SMatrix{out,in}(uview(p,1:(out*in)))
  b = SVector{out}(uview(p,(out*in+1):lastindex(p)))
  res = f.σ.(W*x .+ b)
  function StaticDense_adjoint(ȳ)
    W'*ȳ,vcat(vec(ȳ*res'),ȳ)
  end
  res,StaticDense_adjoint
end
paramlength(f::StaticDense{out,in}) where {out,in} = out*(in + 1)
initial_params(f::StaticDense) = f.initial_params()
