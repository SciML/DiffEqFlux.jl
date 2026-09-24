"""
    CollocationKernel

Abstract interface for kernels used by [`collocate_data`](@ref) when estimating
smoothed state and derivative values from sampled time-series data.

# Interface

Concrete kernels are immutable marker types with zero fields. A kernel implementation
must provide `DiffEqFlux.calckernel(kernel, t)` for an offset `t`, returning the
kernel weight at that offset with the same numeric type as `t` when possible.

Compact-support kernels may implement
`DiffEqFlux.calckernel(kernel, t, abs(t))`; the generic two-argument method handles
the support check and calls the three-argument method only when `abs(t) <= 1`.

# Implementations

The public kernel choices are [`EpanechnikovKernel`](@ref), [`UniformKernel`](@ref),
[`TriangularKernel`](@ref), [`QuarticKernel`](@ref), [`TriweightKernel`](@ref),
[`TricubeKernel`](@ref), [`GaussianKernel`](@ref), [`CosineKernel`](@ref),
[`LogisticKernel`](@ref), [`SigmoidKernel`](@ref), and [`SilvermanKernel`](@ref).

# Examples

```julia
using DiffEqFlux

kernel = TriangularKernel()
du, u = collocate_data(rand(2, 10), range(0, 1; length = 10), kernel)
```
"""
abstract type CollocationKernel end

"""
    EpanechnikovKernel()

Epanechnikov compact-support smoothing kernel for [`collocate_data`](@ref).

# Returns

An immutable [`CollocationKernel`](@ref) with support on `[-1, 1]`.

# Examples

```julia
du, u = collocate_data(rand(2, 10), range(0, 1; length = 10), EpanechnikovKernel())
```
"""
struct EpanechnikovKernel <: CollocationKernel end

"""
    UniformKernel()

Uniform compact-support smoothing kernel for [`collocate_data`](@ref).

# Returns

An immutable [`CollocationKernel`](@ref) with constant weight on `[-1, 1]`.

# Examples

```julia
du, u = collocate_data(rand(2, 10), range(0, 1; length = 10), UniformKernel())
```
"""
struct UniformKernel <: CollocationKernel end

"""
    TriangularKernel()

Triangular compact-support smoothing kernel for [`collocate_data`](@ref).

# Returns

An immutable [`CollocationKernel`](@ref) with linearly decaying weight on `[-1, 1]`.

# Examples

```julia
du, u = collocate_data(rand(2, 10), range(0, 1; length = 10), TriangularKernel())
```
"""
struct TriangularKernel <: CollocationKernel end

"""
    QuarticKernel()

Quartic compact-support smoothing kernel for [`collocate_data`](@ref).

# Returns

An immutable [`CollocationKernel`](@ref) with support on `[-1, 1]`.

# Examples

```julia
du, u = collocate_data(rand(2, 10), range(0, 1; length = 10), QuarticKernel())
```
"""
struct QuarticKernel <: CollocationKernel end

"""
    TriweightKernel()

Triweight compact-support smoothing kernel for [`collocate_data`](@ref).

# Returns

An immutable [`CollocationKernel`](@ref) with support on `[-1, 1]`.

# Examples

```julia
du, u = collocate_data(rand(2, 10), range(0, 1; length = 10), TriweightKernel())
```
"""
struct TriweightKernel <: CollocationKernel end

"""
    TricubeKernel()

Tricube compact-support smoothing kernel for [`collocate_data`](@ref).

# Returns

An immutable [`CollocationKernel`](@ref) with support on `[-1, 1]`.

# Examples

```julia
du, u = collocate_data(rand(2, 10), range(0, 1; length = 10), TricubeKernel())
```
"""
struct TricubeKernel <: CollocationKernel end

"""
    GaussianKernel()

Gaussian smoothing kernel for [`collocate_data`](@ref).

# Returns

An immutable [`CollocationKernel`](@ref) with non-compact Gaussian support.

# Examples

```julia
du, u = collocate_data(rand(2, 10), range(0, 1; length = 10), GaussianKernel())
```
"""
struct GaussianKernel <: CollocationKernel end

"""
    CosineKernel()

Cosine compact-support smoothing kernel for [`collocate_data`](@ref).

# Returns

An immutable [`CollocationKernel`](@ref) with support on `[-1, 1]`.

# Examples

```julia
du, u = collocate_data(rand(2, 10), range(0, 1; length = 10), CosineKernel())
```
"""
struct CosineKernel <: CollocationKernel end

"""
    LogisticKernel()

Logistic smoothing kernel for [`collocate_data`](@ref).

# Returns

An immutable [`CollocationKernel`](@ref) with non-compact logistic support.

# Examples

```julia
du, u = collocate_data(rand(2, 10), range(0, 1; length = 10), LogisticKernel())
```
"""
struct LogisticKernel <: CollocationKernel end

"""
    SigmoidKernel()

Sigmoid smoothing kernel for [`collocate_data`](@ref).

# Returns

An immutable [`CollocationKernel`](@ref) with non-compact sigmoid support.

# Examples

```julia
du, u = collocate_data(rand(2, 10), range(0, 1; length = 10), SigmoidKernel())
```
"""
struct SigmoidKernel <: CollocationKernel end

"""
    SilvermanKernel()

Silverman smoothing kernel for [`collocate_data`](@ref).

# Returns

An immutable [`CollocationKernel`](@ref) with non-compact Silverman support.

# Examples

```julia
du, u = collocate_data(rand(2, 10), range(0, 1; length = 10), SilvermanKernel())
```
"""
struct SilvermanKernel <: CollocationKernel end

"""
    DiffEqFlux.calckernel(kernel, t)

Evaluate a collocation kernel at the normalized offset `t`. This is the developer
extension point for custom [`CollocationKernel`](@ref) implementations.

# Rules

  - Implement either the two-argument method for a kernel with non-compact support,
    or the three-argument method `calckernel(kernel, t, abs_t)` for a kernel supported
    on `[-1, 1]`.
  - Return a scalar with a numeric type compatible with `t`.
  - The generic two-argument method applies the compact-support check before calling
    the three-argument method.
"""
function calckernel(kernel, t::T) where {T}
    abst = abs(t)
    return ifelse(abst > 1, T(0), calckernel(kernel, t, abst))
end
calckernel(::EpanechnikovKernel, t::T, abst::T) where {T} = T(0.75) * (T(1) - t^2)
calckernel(::UniformKernel, t::T, abst::T) where {T} = T(0.5)
calckernel(::TriangularKernel, t::T, abst::T) where {T} = T(1) - abst
calckernel(::QuarticKernel, t::T, abst::T) where {T} = T(15) * (T(1) - t^2)^2 / T(16)
calckernel(::TriweightKernel, t::T, abst::T) where {T} = T(35) * (T(1) - t^2)^3 / T(32)
calckernel(::TricubeKernel, t::T, abst::T) where {T} = T(70) * (T(1) - abst^3)^3 / T(81)
calckernel(::CosineKernel, t::T, abst::T) where {T} = T(π) * cospi(t / T(2)) / T(4)

calckernel(::GaussianKernel, t::T) where {T} = exp(-t^2 / T(2)) / sqrt(T(2) * π)
calckernel(::LogisticKernel, t::T) where {T} = T(1) / (exp(t) + T(2) + exp(-t))
calckernel(::SigmoidKernel, t::T) where {T} = T(2) / (π * (exp(t) + exp(-t)))
function calckernel(::SilvermanKernel, t::T) where {T}
    return sin(abs(t) / T(2) + π / T(4)) * T(0.5) * exp(-abs(t) / sqrt(T(2)))
end

construct_t1(t, tpoints) = hcat(ones(eltype(tpoints), length(tpoints)), tpoints .- t)

function construct_t2(t, tpoints)
    return hcat(ones(eltype(tpoints), length(tpoints)), tpoints .- t, (tpoints .- t) .^ 2)
end

function construct_w(t, tpoints, h, kernel)
    W = @. calckernel((kernel,), ((tpoints - t) / (tpoints[end] - tpoints[begin])) / h) / h
    return Diagonal(W)
end

"""
    u′, u = collocate_data(data, tpoints, kernel = TriangularKernel(), bandwidth=nothing)
    u′, u = collocate_data(data, tpoints, tpoints_sample, interp, args...)

Computes a non-parametrically smoothed estimate of `u'` and `u` given the `data`, where each
column is a snapshot of the timeseries at `tpoints[i]`.

# Arguments

  - `data`: Array of observed state values. For matrix data, each column is one
    observation at the corresponding entry of `tpoints`.
  - `tpoints`: Sample times for `data`.
  - `kernel`: A [`CollocationKernel`](@ref) used for local regression smoothing.
    Defaults to [`TriangularKernel`](@ref).
  - `bandwidth`: Smoothing bandwidth. If `nothing`, a rule-of-thumb bandwidth is used.
  - `tpoints_sample`: Time points where interpolation-based collocation should be sampled.
  - `interp`: DataInterpolations.jl interpolation constructor used by the extension method.
  - `args...`: Additional positional arguments forwarded to the interpolation constructor.

# Returns

  - `u′`: Estimated time derivatives at the requested points.
  - `u`: Smoothed state estimates at the requested points.

# Kernel Choices

The following kernel constructors are provided:

  - EpanechnikovKernel
  - UniformKernel
  - TriangularKernel
  - QuarticKernel
  - TriweightKernel
  - TricubeKernel
  - GaussianKernel
  - CosineKernel
  - LogisticKernel
  - SigmoidKernel
  - SilvermanKernel

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2631937/

Additionally, we can use interpolation methods from
[DataInterpolations.jl](https://github.com/SciML/DataInterpolations.jl) to generate
data from intermediate timesteps. In this case, pass any of the methods like
`QuadraticInterpolation` as `interp`, and the timestamps to sample from as `tpoints_sample`.

# Examples

```julia
using DiffEqFlux

tpoints = range(0, 1; length = 20)
data = reduce(hcat, ([sin(t), cos(t)] for t in tpoints))
du, u = collocate_data(data, tpoints, EpanechnikovKernel())
```
"""
function collocate_data(data, tpoints, kernel = TriangularKernel(), bandwidth = nothing)
    _one = oneunit(first(data))
    _zero = zero(first(data))
    e1 = [_one; _zero]
    e2 = [_zero; _one; _zero]
    n = length(tpoints)
    bandwidth = bandwidth === nothing ?
        (n^(-1 / 5)) * (n^(-3 / 35)) * ((log(n))^(-1 / 16)) : bandwidth

    Wd = similar(data, n, size(data, 1))
    WT1 = similar(data, n, 2)
    WT2 = similar(data, n, 3)
    T2WT2 = similar(data, 3, 3)
    T1WT1 = similar(data, 2, 2)
    x = map(tpoints) do _t
        T1 = construct_t1(_t, tpoints)
        T2 = construct_t2(_t, tpoints)
        W = construct_w(_t, tpoints, bandwidth, kernel)
        mul!(Wd, W, data')
        mul!(WT1, W, T1)
        mul!(WT2, W, T2)
        mul!(T2WT2, T2', WT2)
        mul!(T1WT1, T1', WT1)
        (det(T2WT2) ≈ 0.0 || det(T1WT1) ≈ 0.0) &&
            error("Collocation failed with bandwidth $bandwidth. Please choose a higher bandwidth")
        (e2' * ((T2' * WT2) \ T2')) * Wd, (e1' * ((T1' * WT1) \ T1')) * Wd
    end
    estimated_derivative = mapreduce(xᵢ -> transpose(first(xᵢ)), hcat, x)
    estimated_solution = mapreduce(xᵢ -> transpose(last(xᵢ)), hcat, x)
    return estimated_derivative, estimated_solution
end

@views function collocate_data(
        data::AbstractVector, tpoints::AbstractVector,
        tpoints_sample::AbstractVector, interp, args...
    )
    du, u = collocate_data(reshape(data, 1, :), tpoints, tpoints_sample, interp, args...)
    return du[1, :], u[1, :]
end
