abstract type FastLayer <: Function end

paramlength(f) = 0
initial_params(f) = Flux.destructure(f)[1]

struct FastChain{T<:Tuple} <: FastLayer
  layers::T
  function FastChain(xs...)
    @warn "FastChain is being deprecated in favor of Lux.jl. Lux.jl uses functions with explicit parameters f(u,p) like FastChain, but is fully featured and documented machine learning library. See the Lux.jl documentation for more details."
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
          bias = true, precache = false ,initW = Flux.glorot_uniform, initb = Flux.zeros32)

A Dense layer `activation.(W*x + b)` with input size `in` and output size `out`.
The `activation` function defaults to `identity`, meaning the layer is an affine
function. Initial parameters are taken to match `Flux.Dense`. 'bias' represents b in
the layer and it defaults to true.'precache' is used to preallocate memory for the
intermediate variables calculated during each pass. This avoids heap allocations
in each pass which would otherwise slow down the computation, it defaults to false.

Note that this function has specializations on `tanh` for a slightly faster
adjoint with Zygote.
"""
struct FastDense{F,F2,C} <: FastLayer
  out::Int
  in::Int
  σ::F
  initial_params::F2
  cache :: C
  bias::Bool
  numcols::Int
  function FastDense(in::Integer, out::Integer, σ = identity;
                 bias = true, numcols=1, precache=false, initW = Flux.glorot_uniform, initb = Flux.zeros32)
    temp = ((bias == false) ? vcat(vec(initW(out, in))) : vcat(vec(initW(out, in)),initb(out)))
    initial_params() = temp
    if precache == true
      cache = (
        cols = zeros(Int, 1),
        W = zeros(out, in),
        y = zeros(out, numcols),
        yvec = zeros(out),
        r = zeros(out, numcols),
        zbar = zeros(out, numcols),
        Wbar = zeros(out, in),
        xbar = zeros(in, numcols),
        pbar = if bias == true
            zeros((out*in)+out)
          else
            zeros(out*in)
          end
        )
      else
        cache = nothing
      end
      _σ = NNlib.fast_act(σ)
      new{typeof(_σ), typeof(initial_params), typeof(cache)}(out,in,_σ,initial_params,cache,bias,numcols)
  end
end

# To work with scalars(x::Number)
(f::FastDense)(x::Number,p) = ((f.bias == true) ? (f.σ.(reshape(p[1:(f.out*f.in)],f.out,f.in)*x .+ p[(f.out*f.in+1):end])) : (f.σ.(reshape(p[1:(f.out*f.in)],f.out,f.in)*x)))

ZygoteRules.@adjoint function (f::FastDense)(x::Number,p)
  if typeof(f.cache) <: Nothing
    if !isgpu(p)
      W = @view(p[reshape(1:(f.out*f.in),f.out,f.in)])
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
    y = f.σ.(r)
  else
    if !isgpu(p)
      f.cache.W .= @view(p[reshape(1:(f.out*f.in),f.out,f.in)])
    else
      f.cache.W .= reshape(@view(p[1:(f.out*f.in)]),f.out,f.in)
    end
    mul!(@view(f.cache.r[:,1]), f.cache.W, x)
    if f.bias == true
      # @view(f.cache.r[:,1]) .+= @view(p[(f.out*f.in + 1):end])
      b = @view(p[(f.out*f.in + 1):end])
      @view(f.cache.r[:,1]) .+= b
    end
    f.cache.yvec .= f.σ.(@view(f.cache.r[:,1]))
  end
  function FastDense_adjoint(ȳ)
    if typeof(f.cache) <: Nothing
      if typeof(f.σ) <: typeof(NNlib.tanh_fast)
        zbar = ȳ .* (1 .- y.^2)
      elseif typeof(f.σ) <: typeof(identity)
        zbar = ȳ
      else
        zbar = ȳ .* ForwardDiff.derivative.(f.σ,r)
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
      xb = xbar[1,1]
      nothing,xb,pbar
    else
      if typeof(f.σ) <: typeof(NNlib.tanh_fast)
        @view(f.cache.zbar[:,1]) .= ȳ .* (1 .- (f.cache.yvec).^2)
      elseif typeof(f.σ) <: typeof(identity)
        @view(f.cache.zbar[:,1]) .= ȳ
      else
        @view(f.cache.zbar[:,1]) .= ȳ .* ForwardDiff.derivative.(f.σ, @view(f.cache.r[:,1]))
      end
      mul!(f.cache.Wbar, @view(f.cache.zbar[:,1]), x')
      mul!(@view(f.cache.xbar[:,1]), f.cache.W', @view(f.cache.zbar[:,1]))
      f.cache.pbar .= if f.bias == true
        vec(vcat(vec(f.cache.Wbar),@view(f.cache.zbar[:,1])))# bbar = zbar
      else
        vec(f.cache.Wbar)
      end
      xbar = f.cache.xbar[1,1]
      nothing,xbar,f.cache.pbar
    end
  end
  if typeof(f.cache) <: Nothing
    y,FastDense_adjoint
  else
    f.cache.yvec,FastDense_adjoint
  end
end

# (f::FastDense)(x,p) = f.σ.(reshape(uview(p,1:(f.out*f.in)),f.out,f.in)*x .+ uview(p,(f.out*f.in+1):lastindex(p)))
(f::FastDense)(x::AbstractVector,p) = ((f.bias == true) ? (f.σ.(reshape(p[1:(f.out*f.in)],f.out,f.in)*x .+ p[(f.out*f.in+1):end])) : (f.σ.(reshape(p[1:(f.out*f.in)],f.out,f.in)*x)))

ZygoteRules.@adjoint function (f::FastDense)(x::AbstractVector,p)
  if typeof(f.cache) <: Nothing
    if !isgpu(p)
      W = @view(p[reshape(1:(f.out*f.in),f.out,f.in)])
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
    y = f.σ.(r)
  else
    if !isgpu(p)
      f.cache.W .= @view(p[reshape(1:(f.out*f.in),f.out,f.in)])
    else
      f.cache.W .= reshape(@view(p[1:(f.out*f.in)]),f.out,f.in)
    end
    mul!(@view(f.cache.r[:,1]), f.cache.W, x)
    if f.bias == true
      # @view(f.cache.r[:,1]) .+= @view(p[(f.out*f.in + 1):end])
      b = @view(p[(f.out*f.in + 1):end])
      @view(f.cache.r[:,1]) .+= b
    end
    f.cache.yvec .= f.σ.(@view(f.cache.r[:,1]))
  end
  function FastDense_adjoint(ȳ)
    if typeof(f.cache) <: Nothing
      if typeof(f.σ) <: typeof(NNlib.tanh_fast)
        zbar = ȳ .* (1 .- y.^2)
      elseif typeof(f.σ) <: typeof(identity)
        zbar = ȳ
      else
        zbar = ȳ .* ForwardDiff.derivative.(f.σ,r)
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
    else
      if typeof(f.σ) <: typeof(NNlib.tanh_fast)
        @view(f.cache.zbar[:,1]) .= ȳ .* (1 .- (f.cache.yvec).^2)
      elseif typeof(f.σ) <: typeof(identity)
        @view(f.cache.zbar[:,1]) .= ȳ
      else
        @view(f.cache.zbar[:,1]) .= ȳ .* ForwardDiff.derivative.(f.σ, @view(f.cache.r[:,1]))
      end
      mul!(f.cache.Wbar, @view(f.cache.zbar[:,1]), x')
      mul!(@view(f.cache.xbar[:,1]), f.cache.W', @view(f.cache.zbar[:,1]))
      f.cache.pbar .= if f.bias == true
        vec(vcat(vec(f.cache.Wbar),@view(f.cache.zbar[:,1])))# bbar = zbar
      else
        vec(f.cache.Wbar)
      end
      nothing,@view(f.cache.xbar[:,1]),f.cache.pbar
    end
  end
  if typeof(f.cache) <: Nothing
    y,FastDense_adjoint
  else
    f.cache.yvec,FastDense_adjoint
  end
end

(f::FastDense)(x::AbstractMatrix,p) = ((f.bias == true) ? (f.σ.(reshape(p[1:(f.out*f.in)],f.out,f.in)*x .+ p[(f.out*f.in+1):end])) : (f.σ.(reshape(p[1:(f.out*f.in)],f.out,f.in)*x)))

ZygoteRules.@adjoint function (f::FastDense)(x::AbstractMatrix,p)
  if typeof(f.cache) <: Nothing
    if !isgpu(p)
      W = @view(p[reshape(1:(f.out*f.in),f.out,f.in)])
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
    y = f.σ.(r)
  else
    if !isgpu(p)
      f.cache.W .= @view(p[reshape(1:(f.out*f.in),f.out,f.in)])
    else
      f.cache.W .= reshape(@view(p[1:(f.out*f.in)]),f.out,f.in)
    end
    f.cache.cols[1] = size(x)[2]
    mul!(@view(f.cache.r[:,1:f.cache.cols[1]]), f.cache.W, x)
    if f.bias == true
      @view(f.cache.r[:,1:f.cache.cols[1]]) .+= @view(p[(f.out*f.in + 1):end])
    end
    @view(f.cache.y[:,1:f.cache.cols[1]]) .= f.σ.(@view(f.cache.r[:,1:f.cache.cols[1]]))
  end
  function FastDense_adjoint(ȳ)
    if typeof(f.cache) <: Nothing
      if typeof(f.σ) <: typeof(NNlib.tanh_fast)
        zbar = ȳ .* (1 .- y.^2)
      elseif typeof(f.σ) <: typeof(identity)
        zbar = ȳ
      else
        zbar = ȳ .* ForwardDiff.derivative.(f.σ,r)
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
    else
      if typeof(f.σ) <: typeof(NNlib.tanh_fast)
        @view(f.cache.zbar[:,1:f.cache.cols[1]]) .= ȳ .* (1 .- @view(f.cache.y[:,1:f.cache.cols[1]]).^2)
      elseif typeof(f.σ) <: typeof(identity)
        @view(f.cache.zbar[:,1:f.cache.cols[1]]) .= ȳ
      else
        @view(f.cache.zbar[:,1:f.cache.cols[1]]) .= ȳ .* ForwardDiff.derivative.(f.σ, @view(f.cache.r[:,1:f.cache.cols[1]]))
      end
      mul!(f.cache.Wbar, @view(f.cache.zbar[:,1:f.cache.cols[1]]), x')
      mul!(@view(f.cache.xbar[:,1:f.cache.cols[1]]), f.cache.W', @view(f.cache.zbar[:,1:f.cache.cols[1]]))
      f.cache.pbar .= if f.bias == true
        vec(vcat(vec(f.cache.Wbar),sum(@view(f.cache.zbar[:,1:f.cache.cols[1]]),dims=2)))# bbar = zbar
      else
        vec(f.cache.Wbar)
      end
      nothing,@view(f.cache.xbar[:,1:f.cache.cols[1]]),f.cache.pbar
    end
  end
  if typeof(f.cache) <: Nothing
    y,FastDense_adjoint
  elseif f.numcols == f.cache.cols[1]
    f.cache.y,FastDense_adjoint
  else
    @view(f.cache.y[:,1:f.cache.cols[1]]),FastDense_adjoint
  end
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
  function StaticDense_adjoint(ȳ)
    if typeof(f.σ) <: typeof(tanh)
      σbar = 1 .- y.^2
    else
      σbar = ForwardDiff.derivative.(f.σ,r)
    end
    if typeof(ȳ) <: AbstractMatrix
      error("StaticDense only supports vector data")
    end
    zbar = SVector{out}(ȳ) .* σbar
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
