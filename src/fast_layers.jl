paramlength(f) = 0

struct FastChain{T<:Tuple}
  layers::T
  function FastChain(xs...)
    layers = getfunc.(xs)
    ps = vcat(getparams.(xs)...)
    new{typeof(layers)}(layers),ps
  end
end
getfunc(x) = x
getfunc(x::Tuple) = first(x)
getparams(x) = Float32[]
getparams(x::Tuple) = last(x)

applychain(::Tuple{}, x, p) = x
applychain(fs::Tuple, x, p) = applychain(Base.tail(fs), first(fs)(x,p[1:paramlength(first(fs))]), p[(paramlength(first(fs))+1):end])
(c::FastChain)(x,p) = applychain(c.layers, x, p)

struct FastDense{F} <: Function
  out::Int
  in::Int
  σ::F
  function FastDense(in::Integer, out::Integer, σ = identity;
                 initW = Flux.glorot_uniform, initb = zeros)
    new{typeof(σ)}(out,in,σ),vcat(vec(initW(out, in)),initb(out))
  end
end
(f::FastDense)(x,p) = f.σ.(reshape(p[1:(f.out*f.in)],f.out,f.in)*x .+ p[(f.out*f.in+1):end])
paramlength(f::FastDense) = f.out*(f.in + 1)

struct StaticDense{out,in,F} <: Function
  σ::F
  function FastDense(in::Integer, out::Integer, σ = identity;
                 initW = Flux.glorot_uniform, initb = zeros)
    new{out,in,typeof(σ)}(σ),vcat(vec(initW(out, in)),initb(out))
  end
end
(f::StaticDense{out,in})(x,p) where {out,in} = SMatrix*x + SVector{out}(p[(f.out*f.in+1):end])
paramlength(f::StaticDense{out,in}) where {out,in} = out*(in + 1)
