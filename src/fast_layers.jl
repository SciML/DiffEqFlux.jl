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
          bias = true ,initW = Flux.glorot_uniform, initb = Flux.zeros32)

A Dense layer `activation.(W*x + b)` with input size `in` and output size `out`.
The `activation` function defaults to `identity`, meaning the layer is an affine
function. Initial parameters are taken to match `Flux.Dense`. 'bias' represents b in
the layer and it defaults to true.

Note that this function has specializations on `tanh` for a slightly faster
adjoint with Zygote.
"""
struct FastDense{F,F2} <: FastLayer
  out::Int
  in::Int
  σ::F
  initial_params::F2
  bias::Bool
  function FastDense(in::Integer, out::Integer, σ = identity;
                 bias = true, initW = Flux.glorot_uniform, initb = Flux.zeros32)
    temp = ((bias == false) ? vcat(vec(initW(out, in))) : vcat(vec(initW(out, in)),initb(out)))
    initial_params() = temp
    new{typeof(σ),typeof(initial_params)}(out,in,σ,initial_params,bias)
  end
end

# (f::FastDense)(x,p) = f.σ.(reshape(uview(p,1:(f.out*f.in)),f.out,f.in)*x .+ uview(p,(f.out*f.in+1):lastindex(p)))
(f::FastDense)(x,p) = ((f.bias == true) ? (f.σ.(reshape(p[1:(f.out*f.in)],f.out,f.in)*x .+ p[(f.out*f.in+1):end])) : (f.σ.(reshape(p[1:(f.out*f.in)],f.out,f.in)*x)))

ZygoteRules.@adjoint function (f::FastDense)(x,p)
  if !isgpu(p)
    W = @view p[reshape(1:(f.out*f.in),f.out,f.in)]
  else
    W = reshape(@view(p[1:(f.out*f.in)]),f.out,f.in)
  end

  if f.bias == true
    b = p[(f.out*f.in + 1):end]
    r = W*x .+ b
    ifgpufree(b)
  else
    r = W*x
  end

  # if typeof(x) <: AbstractVector
  #   r = p[(f.out*f.in+1):end]
  #   mul!(r,W,x,one(eltype(x)),one(eltype(x)))
  # else
  #   b = @view p[(f.out*f.in+1):end]
  #   r = reshape(repeat(b,outer=size(x,2)),length(b),size(x,2))
  #   mul!(r,W,x,one(eltype(x)),one(eltype(x)))
  # end

  y = f.σ.(r)

  function FastDense_adjoint(ȳ)
    if typeof(f.σ) <: typeof(tanh)
      zbar = ȳ .* (1 .- y.^2)
    elseif typeof(f.σ) <: typeof(identity)
      zbar = ȳ
    else
      zbar = ȳ .* ForwardDiff.derivative.(f.σ,r)
    end
    Wbar = zbar * x'
    bbar = zbar
    xbar = W' * zbar
    pbar = if f.bias == true
        tmp = typeof(bbar) <: AbstractVector ?
                         vec(vcat(vec(Wbar),bbar)) :
                         vec(vcat(vec(Wbar),sum(bbar,dims=2)))
        ifgpufree(bbar)
        tmp
    else
        vec(Wbar)
    end
    ifgpufree(Wbar)
    ifgpufree(r)
    nothing,xbar,pbar
  end
  y,FastDense_adjoint
end
paramlength(f::FastDense) = f.out*(f.in+f.bias)
initial_params(f::FastDense) = f.initial_params()

"""
StaticDense(in,out,activation=identity;
          initW = Flux.glorot_uniform, initb = Flux.zeros32)

A Dense layer `activation.(W*x + b)` with input size `in` and output size `out`.
The `activation` function defaults to `identity`, meaning the layer is an affine
function. Initial parameters are taken to match `Flux.Dense`. The internal
calculations are done with `StaticArrays` for extra speed for small linear
algebra operations. Should only be used for input/output sizes of approximately
16 or less. 'bias' represents bias(b) in the dense layer and it defaults to true.

Note that this function has specializations on `tanh` for a slightly faster
adjoint with Zygote.
"""
struct StaticDense{out,in,bias,F,F2} <: FastLayer
  σ::F
  initial_params::F2
  function StaticDense(in::Integer, out::Integer, σ = identity;
                 bias::Bool = true, initW = Flux.glorot_uniform, initb = Flux.zeros32)
    temp = ((bias == true) ? vcat(vec(initW(out, in)),initb(out)) : vcat(vec(initW(out, in))))
    initial_params() = temp
    new{out,in,bias,typeof(σ),typeof(initial_params)}(σ,initial_params)
  end
end

function param2Wb(f::StaticDense{out,in,bias}, p) where {out,in,bias}
  if bias == true
    _W, _b = @views p[1:(out*in)], p[(out*in+1):end]
    W = @inbounds convert(SMatrix{out,in},_W)
    b = @inbounds SVector{out}(_b)
  return W, b
  else
    _W = @view p[1:(out*in)]
    W = @inbounds convert(SMatrix{out,in},_W)
    return W
  end
end
function (f::StaticDense{out,in,bias})(x,p) where {out,in,bias}
  if bias == true
    W, b = param2Wb(f, p)
    return f.σ.(W*x .+ b)
  else
    W = param2Wb(f,p)
    return f.σ.(W*x)
  end
end
ZygoteRules.@adjoint function (f::StaticDense{out,in,bias})(x,p) where {out,in,bias}
  if bias == true
    W, b = param2Wb(f, p)
    r = W*x .+ b
  else
    W = param2Wb(f,p)
    r = W*x
  end
  y = f.σ.(r)
  function StaticDense_adjoint(ȳ)
    if typeof(f.σ) <: typeof(tanh)
      σbar = 1 .- y.^2
    else
      σbar = ForwardDiff.derivative.(f.σ,r)
    end
    if typeof(ȳ) <: AbstractMatrix
      error("StaticDense only supports vector data")
    end
    zbar = SVector{out}(ȳ) .* σbar
    Wbar = zbar * SVector{in}(x)'
    bbar = zbar
    xbar = W' * zbar
    pbar = if bias == true
        tmp = typeof(bbar) <: AbstractVector ?
                         vec(vcat(vec(Wbar),bbar)) :
                         vec(vcat(vec(Wbar),sum(bbar,dims=2)))
        tmp
    else
        vec(Wbar)
    end
    xbar_out = x isa SArray ? xbar : adapt(typeof(x),xbar)
    pbar_out = p isa SArray ? pbar : adapt(typeof(p),pbar)
    nothing,xbar_out,pbar_out
  end
  y,StaticDense_adjoint
end
paramlength(f::StaticDense{out,in,bias}) where {out,in,bias} = out*(in + bias)
initial_params(f::StaticDense) = f.initial_params()

# Override FastDense to exclude the branch from the check
function Cassette.overdub(ctx::DiffEqSensitivity.HasBranchingCtx, f::FastDense, x, p)
    y = reshape(p[1:(f.out*f.in)],f.out,f.in)*x
    Cassette.@overdub ctx f.σ.(y)
end

function Cassette.overdub(ctx::DiffEqSensitivity.HasBranchingCtx, f::StaticDense{out,in,bias}, x, p) where {out,in,bias}
    y = reshape(p[1:(out*in)],out,in)*x
    Cassette.@overdub ctx f.σ.(y)
end
