abstract type NeuralDELayer <: LuxCore.AbstractExplicitContainerLayer{(:model,)} end
abstract type NeuralSDELayer <: LuxCore.AbstractExplicitContainerLayer{(:drift,:diffusion,)} end
basic_tgrad(u,p,t) = zero(u)
basic_dde_tgrad(u,h,p,t) = zero(u)

"""
Constructs a continuous-time recurrant neural network, also known as a neural
ordinary differential equation (neural ODE), with a fast gradient calculation
via adjoints [1]. At a high level this corresponds to solving the forward
differential equation, using a second differential equation that propagates the
derivatives of the loss backwards in time.

```julia
NeuralODE(model,tspan,alg=nothing,args...;kwargs...)
```

Arguments:

- `model`: A Flux.Chain or Lux.AbstractExplicitLayer neural network that defines the ̇x.
- `tspan`: The timespan to be solved on.
- `alg`: The algorithm used to solve the ODE. Defaults to `nothing`, i.e. the
  default algorithm from DifferentialEquations.jl.
- `sensealg`: The choice of differentiation algorthm used in the backpropogation.
  Defaults to an adjoint method. See
  the [Local Sensitivity Analysis](https://docs.sciml.ai/DiffEqDocs/stable/analysis/sensitivity/)
  documentation for more details.
- `kwargs`: Additional arguments splatted to the ODE solver. See the
  [Common Solver Arguments](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/)
  documentation for more details.

References:

[1] Pontryagin, Lev Semenovich. Mathematical theory of optimal processes. CRC press, 1987.

"""
struct NeuralODE{M,P,RE,T,A,K} <: NeuralDELayer
    model::M
    p::P
    st::NamedTuple
    re::RE
    tspan::T
    args::A
    kwargs::K
end

function NeuralODE(model::LuxCore.AbstractExplicitLayer,tspan,args...;p=nothing,st=NamedTuple(),kwargs...)
  re = nothing
  NeuralODE{typeof(model),typeof(p),typeof(re),
      typeof(tspan),typeof(args),typeof(kwargs)}(
      model,p,st,re,tspan,args,kwargs)
end

function (n::NeuralODE{M})(x,p=n.p,st=n.st) where {M<:LuxCore.AbstractExplicitLayer}
  function dudt_(u,p,t;st=st)
    u_, st = n.model(u,p,st)
    return u_
  end

  ff = ODEFunction{false}(dudt_,tgrad=basic_tgrad)
  prob = ODEProblem{false}(ff,x,n.tspan,p)
  sense = InterpolatingAdjoint(autojacvec=ZygoteVJP())
  return solve(prob,n.args...;sensealg=sense,n.kwargs...), st
end

"""
Constructs a neural stochastic differential equation (neural SDE) with diagonal noise.

```julia
NeuralDSDE(drift,diffusion,tspan,alg=nothing,args...;
           sensealg=TrackerAdjoint(),kwargs...)
```

Arguments:

- `drift`: A Flux.Chain or Lux.AbstractExplicitLayer neural network that defines the drift function.
- `diffusion`: A Flux.Chain or Lux.AbstractExplicitLayer neural network that defines the diffusion function.
  Should output a vector of the same size as the input.
- `tspan`: The timespan to be solved on.
- `alg`: The algorithm used to solve the ODE. Defaults to `nothing`, i.e. the
  default algorithm from DifferentialEquations.jl.
- `sensealg`: The choice of differentiation algorthm used in the backpropogation.
  Defaults to using reverse-mode automatic differentiation via Tracker.jl
- `kwargs`: Additional arguments splatted to the ODE solver. See the
  [Common Solver Arguments](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/)
  documentation for more details.

"""
struct NeuralDSDE{M,P,RE,M2,RE2,T,A,K} <: NeuralSDELayer
    p::P
    st::NamedTuple
    len::Int
    drift::M
    re1::RE
    diffusion::M2
    re2::RE2
    tspan::T
    args::A
    kwargs::K
end

function NeuralDSDE(drift::LuxCore.AbstractExplicitLayer,diffusion::LuxCore.AbstractExplicitLayer,tspan,args...;
                    p=nothing,st=NamedTuple(drift=NamedTuple(),diffusion=NamedTuple()),kwargs...)
  re1 = nothing
  re2 = nothing
  NeuralDSDE{typeof(drift),typeof(p),typeof(re1),typeof(diffusion),typeof(re2),
      typeof(tspan),typeof(args),typeof(kwargs)}(
      p,st,1,drift,re1,diffusion,re2,tspan,args,kwargs)
end

function (n::NeuralDSDE{M})(x,p=n.p,st=n.st) where {M<:LuxCore.AbstractExplicitLayer}
    st1 = st.drift
    st2 = st.diffusion
    function dudt_(u,p,t;st=st1)
      u_, st = n.drift(u,p.drift,st)
      return u_
    end
    function g(u,p,t;st=st2)
      u_, st = n.diffusion(u,p.diffusion,st)
      return u_
    end

    ff = SDEFunction{false}(dudt_,g,tgrad=basic_tgrad)
    prob = SDEProblem{false}(ff,g,x,n.tspan,p)
    return solve(prob,n.args...;sensealg=BacksolveAdjoint(),n.kwargs...), (drift = st1, diffusion = st2)
end

"""
Constructs a neural stochastic differential equation (neural SDE).

```julia
NeuralSDE(drift,diffusion,tspan,nbrown,alg=nothing,args...;
          sensealg=TrackerAdjoint(),kwargs...)
```

Arguments:

- `drift`: A Flux.Chain or Lux.AbstractExplicitLayer neural network that defines the drift function.
- `diffusion`: A Flux.Chain or Lux.AbstractExplicitLayer neural network that defines the diffusion function.
  Should output a matrix that is nbrown x size(x,1).
- `tspan`: The timespan to be solved on.
- `nbrown`: The number of Brownian processes
- `alg`: The algorithm used to solve the ODE. Defaults to `nothing`, i.e. the
  default algorithm from DifferentialEquations.jl.
- `sensealg`: The choice of differentiation algorthm used in the backpropogation.
  Defaults to using reverse-mode automatic differentiation via Tracker.jl
- `kwargs`: Additional arguments splatted to the ODE solver. See the
  [Common Solver Arguments](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/)
  documentation for more details.

"""
struct NeuralSDE{P,M,RE,M2,RE2,T,A,K} <: NeuralSDELayer
    p::P
    st::NamedTuple
    len::Int
    drift::M
    re1::RE
    diffusion::M2
    re2::RE2
    tspan::T
    nbrown::Int
    args::A
    kwargs::K
end

function NeuralSDE(drift::LuxCore.AbstractExplicitLayer, diffusion::LuxCore.AbstractExplicitLayer,tspan,nbrown,args...;
                   p=nothing,st=NamedTuple(drift=NamedTuple(),diffusion=NamedTuple()),kwargs...)
  re1 = nothing
  re2 = nothing
  NeuralSDE{typeof(p),typeof(drift),typeof(re1),typeof(diffusion),typeof(re2),
      typeof(tspan),typeof(args),typeof(kwargs)}(
      p,st,1,drift,re1,diffusion,re2,tspan,nbrown,args,kwargs)
end

function (n::NeuralSDE{P,M})(x,p=n.p,st=n.st) where {P,M<:LuxCore.AbstractExplicitLayer}
    st1 = st.drift
    st2 = st.diffusion
    function dudt_(u,p,t;st=st1)
        u_, st = n.drift(u,p.drift,st)
        return u_
    end
    function g(u,p,t;st=st2)
        u_, st = n.diffusion(u,p.diffusion,st)
        return u_
    end

    ff = SDEFunction{false}(dudt_,g,tgrad=basic_tgrad)
    prob = SDEProblem{false}(ff,g,x,n.tspan,p,noise_rate_prototype=zeros(Float32,length(x),n.nbrown))
    return solve(prob,n.args...;sensealg=BacksolveAdjoint(),n.kwargs...), (drift = st1, diffusion = st2)
end

"""
Constructs a neural delay differential equation (neural DDE) with constant
delays.

```julia
NeuralCDDE(model,tspan,hist,lags,alg=nothing,args...;
          sensealg=TrackerAdjoint(),kwargs...)
```

Arguments:

- `model`: A Flux.Chain or Lux.AbstractExplicitLayer neural network that defines the derivative function.
  Should take an input of size `[x;x(t-lag_1);...;x(t-lag_n)]` and produce and
  output shaped like `x`.
- `tspan`: The timespan to be solved on.
- `hist`: Defines the history function `h(t)` for values before the start of the
  integration.
- `lags`: Defines the lagged values that should be utilized in the neural network.
- `alg`: The algorithm used to solve the ODE. Defaults to `nothing`, i.e. the
  default algorithm from DifferentialEquations.jl.
- `sensealg`: The choice of differentiation algorthm used in the backpropogation.
  Defaults to using reverse-mode automatic differentiation via Tracker.jl
- `kwargs`: Additional arguments splatted to the ODE solver. See the
  [Common Solver Arguments](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/)
  documentation for more details.

"""
Unsupported_NeuralCDDE_pairing_message = """
                                         NeuralCDDE can only be instantiated with a Flux chain
                                         """

struct Unsupported_pairing <:Exception
  msg::Any
end

function Base.showerror(io::IO, e::Unsupported_pairing)
  println(io, e.msg)
end

struct NeuralCDDE{P,M,RE,H,L,T,A,K} <: NeuralDELayer
    p::P
    st::NamedTuple
    model::M
    re::RE
    hist::H
    lags::L
    tspan::T
    args::A
    kwargs::K
end

function NeuralCDDE(model::LuxCore.AbstractExplicitLayer,tspan,hist,lags,args...;p = nothing,st=NamedTuple(),kwargs...)
  # throw(Unsupported_pairing(Unsupported_NeuralCDDE_pairing_message))
  re = nothing
  NeuralCDDE{typeof(p),typeof(model),typeof(re),typeof(hist),typeof(lags),
      typeof(tspan),typeof(args),typeof(kwargs)}(
      p,st,model,re,hist,lags,tspan,args,kwargs)
end

function (n::NeuralCDDE)(x,p=n.p,st=n.st)
    function dudt_(u,h,p,t;st=st)
        _u = vcat(u,(h(p,t-lag) for lag in n.lags)...)
        n.model(_u, p, st)
    end
    ff = DDEFunction{false}(dudt_,tgrad=basic_dde_tgrad)
    prob = DDEProblem{false}(ff,x,n.hist,n.tspan,p,constant_lags = n.lags)
    solve(prob,n.args...;sensealg=TrackerAdjoint(),n.kwargs...)
end

"""
Constructs a neural differential-algebraic equation (neural DAE).

```julia
NeuralDAE(model,constraints_model,tspan,alg=nothing,args...;
          sensealg=TrackerAdjoint(),kwargs...)
```

Arguments:

- `model`: A Flux.Chain or Lux.AbstractExplicitLayer neural network that defines the derivative function.
  Should take an input of size `x` and produce the residual of `f(dx,x,t)`
  for only the differential variables.
- `constraints_model`: A function `constraints_model(u,p,t)` for the fixed
  constaints to impose on the algebraic equations.
- `tspan`: The timespan to be solved on.
- `alg`: The algorithm used to solve the ODE. Defaults to `nothing`, i.e. the
  default algorithm from DifferentialEquations.jl.
- `sensealg`: The choice of differentiation algorthm used in the backpropogation.
  Defaults to using reverse-mode automatic differentiation via Tracker.jl
- `kwargs`: Additional arguments splatted to the ODE solver. See the
  [Common Solver Arguments](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/)
  documentation for more details.

"""
struct NeuralDAE{P,M,M2,D,RE,T,DV,A,K} <: NeuralDELayer
    model::M
    constraints_model::M2
    p::P
    st::NamedTuple
    du0::D
    re::RE
    tspan::T
    differential_vars::DV
    args::A
    kwargs::K
end

function NeuralDAE(model::LuxCore.AbstractExplicitLayer,constraints_model,tspan,du0=nothing,args...;p=nothing,st=NamedTuple(),differential_vars=nothing,kwargs...)
  re = nothing

  NeuralDAE{typeof(p),typeof(model),typeof(constraints_model),
      typeof(du0),typeof(re),typeof(tspan),
      typeof(differential_vars),typeof(args),typeof(kwargs)}(
      model,constraints_model,p,st,du0,re,tspan,differential_vars,
      args,kwargs)
end

function (n::NeuralDAE{P,M})(x,p=n.p,st=n.st) where {P,M<:LuxCore.AbstractExplicitLayer}
  du0 = n.du0
  function f(du,u,p,t;st=st)
      nn_out, st = n.model(vcat(u,du),p,st)
      alg_out = n.constraints_model(u,p,t)
      iter_nn = 0
      iter_consts = 0
      map(n.differential_vars) do isdiff
          if isdiff
              iter_nn += 1
              nn_out[iter_nn]
          else
              iter_consts += 1
              alg_out[iter_consts]
          end
      end
  end
  prob = DAEProblem{false}(f,du0,x,n.tspan,p,differential_vars=n.differential_vars)
  return solve(prob,n.args...;sensealg=TrackerAdjoint(),n.kwargs...), st
end

"""
Constructs a physically-constrained continuous-time recurrant neural network,
also known as a neural differential-algebraic equation (neural DAE), with a
mass matrix and a fast gradient calculation via adjoints [1]. The mass matrix
formulation is:

```math
Mu' = f(u,p,t)
```

where `M` is semi-explicit, i.e. singular with zeros for rows corresponding to
the constraint equations.

```julia
NeuralODEMM(model,constraints_model,tspan,mass_matrix,alg=nothing,args...;kwargs...)
```

Arguments:

- `model`: A Flux.Chain or Lux.AbstractExplicitLayer neural network that defines the ̇`f(u,p,t)`
- `constraints_model`: A function `constraints_model(u,p,t)` for the fixed
  constaints to impose on the algebraic equations.
- `tspan`: The timespan to be solved on.
- `mass_matrix`: The mass matrix associated with the DAE
- `alg`: The algorithm used to solve the ODE. Defaults to `nothing`, i.e. the
  default algorithm from DifferentialEquations.jl. This method requires an
  implicit ODE solver compatible with singular mass matrices. Consult the
  [DAE solvers](https://docs.sciml.ai/DiffEqDocs/stable/solvers/dae_solve/) documentation for more details.
- `sensealg`: The choice of differentiation algorthm used in the backpropogation.
  Defaults to an adjoint method. See
  the [Local Sensitivity Analysis](https://docs.sciml.ai/DiffEqDocs/stable/analysis/sensitivity/)
  documentation for more details.
- `kwargs`: Additional arguments splatted to the ODE solver. See the
  [Common Solver Arguments](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/)
  documentation for more details.

"""
struct NeuralODEMM{M,M2,P,RE,T,MM,A,K} <: NeuralDELayer
    model::M
    constraints_model::M2
    p::P
    st::NamedTuple
    re::RE
    tspan::T
    mass_matrix::MM
    args::A
    kwargs::K
end

function NeuralODEMM(model::LuxCore.AbstractExplicitLayer,constraints_model,tspan,mass_matrix,args...;
                      p=nothing,st=NamedTuple(),kwargs...)
  re = nothing
  NeuralODEMM{typeof(model),typeof(constraints_model),typeof(p),typeof(re),
      typeof(tspan),typeof(mass_matrix),typeof(args),typeof(kwargs)}(
      model,constraints_model,p,st,re,tspan,mass_matrix,args,kwargs)
end

function (n::NeuralODEMM{M})(x,p=n.p,st=n.st) where {M<:LuxCore.AbstractExplicitLayer}
  function f(u,p,t;st=st)
      nn_out,st = n.model(u,p,st)
      alg_out = n.constraints_model(u,p,t)
      return vcat(nn_out,alg_out)
  end
  dudt_= ODEFunction{false}(f;mass_matrix=n.mass_matrix,tgrad=basic_tgrad)
  prob = ODEProblem{false}(dudt_,x,n.tspan,p)

  sense = InterpolatingAdjoint(autojacvec=ZygoteVJP())
  return solve(prob,n.args...;sensealg=sense,n.kwargs...), st
end

"""
Constructs an Augmented Neural Differential Equation Layer.

```julia
AugmentedNDELayer(nde, adim::Int)
```

Arguments:

- `nde`: Any Neural Differential Equation Layer
- `adim`: The number of dimensions the initial conditions should be lifted

References:

[1] Dupont, Emilien, Arnaud Doucet, and Yee Whye Teh. "Augmented neural ODEs." In Proceedings of the 33rd International Conference on Neural Information Processing Systems, pp. 3140-3150. 2019.

"""
abstract type AugmentedNDEType <: LuxCore.AbstractExplicitContainerLayer{(:nde,)} end
struct AugmentedNDELayer{DE<:Union{NeuralDELayer,NeuralSDELayer}} <: AugmentedNDEType
    nde::DE
    adim::Int
end

(ande::AugmentedNDELayer)(x, args...) = ande.nde(augment(x, ande.adim), args...)

augment(x::AbstractVector{S}, augment_dim::Int) where S =
    cat(x, zeros(S, (augment_dim,)), dims = 1)

augment(x::AbstractArray{S, T}, augment_dim::Int) where {S, T} =
    cat(x, zeros(S, (size(x)[1:(T - 2)]..., augment_dim, size(x, T))), dims = T - 1)

Base.getproperty(ande::AugmentedNDELayer, sym::Symbol) =
    hasproperty(ande, sym) ? getfield(ande, sym) : getfield(ande.nde, sym)

abstract type HelperLayer <: Function end

"""
Constructs a Dimension Mover Layer.

```julia
DimMover(from, to)
```
"""
struct DimMover <: HelperLayer
    from::Integer
    to::Integer
end

function (dm::DimMover)(x)
    @assert !iszero(dm.from)
    @assert !iszero(dm.to)

    from = dm.from > 0 ? dm.from : (length(size(x)) + 1 + dm.from)
    to = dm.to > 0 ? dm.to : (length(size(x)) + 1 + dm.to)

    cat(eachslice(x; dims=from)...; dims=to)
end

"""
We can have Flux's conventional order (data, channel, batch)
by using it as the last layer of `Flux.Chain` to swap the batch-index and the time-index of the Neural DE's output.
considering that each time point is a channel.

```julia
FluxBatchOrder = DimMover(-2, -1)
```
"""
const FluxBatchOrder = DimMover(-2, -1)
