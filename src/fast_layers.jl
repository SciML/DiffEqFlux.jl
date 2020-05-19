abstract type FastLayer <: Function end

paramlength(f) = 0
initial_params(f) = Float32[]
initial_params(f::Chain) = Flux.destructure(f)[1]

struct FastChain{T<:Tuple} <: FastLayer
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
paramlength(c::FastChain) = sum(paramlength(x) for x in c.layers)
initial_params(c::FastChain) = vcat(initial_params.(c.layers)...)

"""
FastDense(in,out,activation=identity;
          initW = Flux.glorot_uniform, initb = Flux.zeros)

A Dense layer `activation.(W*x + b)` with input size `in` and output size `out`.
The `activation` function defaults to `identity`, meaning the layer is an affine
function. Initial parameters are taken to match `Flux.Dense`.

Note that this function has specializations on `tanh` for a slightly faster
adjoint with Zygote.
"""
struct FastDense{F,F2} <: FastLayer
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
ZygoteRules.@adjoint function (f::FastDense)(x,p)
  @static if VERSION >= v"1.5"
    W = @view p[reshape(1:(f.out*f.in),f.out,f.in)]
  else
    W = p[reshape(1:(f.out*f.in),f.out,f.in)]
  end

  if typeof(x) <: AbstractVector
  r = p[(f.out*f.in+1):end]
  mul!(r,W,x,one(eltype(x)),one(eltype(x)))
  else
    b = p[(f.out*f.in+1):end]
    r = W*x .+ b
  end

  y = f.σ.(r)
  function FastDense_adjoint(ȳ)
    if typeof(f.σ) <: typeof(tanh)
      zbar = ȳ .* (1 .- y.^2)
    else
      zbar = ȳ .* ForwardDiff.derivative.(f.σ,r)
    end
    Wbar = zbar * x'
    bbar = zbar
    xbar = W' * zbar
    pbar = typeof(bbar) <: AbstractVector ?
                             vec(vcat(vec(Wbar),bbar)) :
                             vec(vcat(vec(Wbar),sum(bbar,dims=2)))
    nothing,xbar,pbar
  end
  y,FastDense_adjoint
end
paramlength(f::FastDense) = f.out*(f.in + 1)
initial_params(f::FastDense) = f.initial_params()

"""
StaticDense(in,out,activation=identity;
          initW = Flux.glorot_uniform, initb = Flux.zeros)

A Dense layer `activation.(W*x + b)` with input size `in` and output size `out`.
The `activation` function defaults to `identity`, meaning the layer is an affine
function. Initial parameters are taken to match `Flux.Dense`. The internal
calculations are done with `StaticArrays` for extra speed for small linear
algebra operations. Should only be used for input/output sizes of approximately
16 or less.

Note that this function has specializations on `tanh` for a slightly faster
adjoint with Zygote.
"""
struct StaticDense{out,in,F,F2} <: FastLayer
  σ::F
  initial_params::F2
  function StaticDense(in::Integer, out::Integer, σ = identity;
                 initW = Flux.glorot_uniform, initb = Flux.zeros)
    initial_params() = vcat(vec(initW(out, in)),initb(out))
    new{out,in,typeof(σ),typeof(initial_params)}(σ,initial_params)
  end
end

function param2Wb(f::StaticDense{out,in}, p) where {out,in}
  _W, _b = @views p[1:(out*in)], p[(out*in+1):end]
  W = @inbounds convert(SMatrix{out,in},_W)
  b = @inbounds SVector{out}(_b)
  return W, b
end
function (f::StaticDense{out,in})(x,p) where {out,in}
  W, b = param2Wb(f, p)
  f.σ.(W*x .+ b)
end
ZygoteRules.@adjoint function (f::StaticDense{out,in})(x,p) where {out,in}
  W, b = param2Wb(f, p)
  r = W*x .+ b
  y = f.σ.(r)
  function StaticDense_adjoint(ȳ)
    if typeof(f.σ) <: typeof(tanh)
      zbar = SVector{out}(ȳ) .* 1 .- y.^2
    else
      zbar = SVector{out}(ȳ) .* ForwardDiff.derivative.(f.σ,r)
    end
    if typeof(ȳ) <: AbstractMatrix
      error("StaticDense only supports vector data")
    end
    Wbar = zbar * SVector{in}(x)'
    bbar = zbar
    xbar = W' * zbar
    pbar = typeof(bbar) <: AbstractVector ?
                             vec(vcat(vec(Wbar),bbar)) :
                             vec(vcat(vec(Wbar),sum(bbar,dims=2)))
    xbar_out = x isa SArray ? xbar : adapt(typeof(x),xbar)
    pbar_out = p isa SArray ? pbar : adapt(typeof(p),pbar)
    nothing,xbar_out,pbar_out
  end
  y,StaticDense_adjoint
end
paramlength(f::StaticDense{out,in}) where {out,in} = out*(in + 1)
initial_params(f::StaticDense) = f.initial_params()
