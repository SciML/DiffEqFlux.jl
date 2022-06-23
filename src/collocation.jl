abstract type CollocationKernel end
struct EpanechnikovKernel <: CollocationKernel end
struct UniformKernel <: CollocationKernel end
struct TriangularKernel <: CollocationKernel end
struct QuarticKernel <: CollocationKernel end
struct TriweightKernel <: CollocationKernel end
struct TricubeKernel <: CollocationKernel end
struct GaussianKernel <: CollocationKernel end
struct CosineKernel <: CollocationKernel end
struct LogisticKernel <: CollocationKernel end
struct SigmoidKernel <: CollocationKernel end
struct SilvermanKernel <: CollocationKernel end

function calckernel(::EpanechnikovKernel, t)
    if abs(t) > 1
        return 0
    else
        return 0.75 * (1 - t^2)
    end
end

function calckernel(::UniformKernel, t)
    if abs(t) > 1
        return 0
    else
        return 0.5
    end
end

function calckernel(::TriangularKernel, t)
    if abs(t) > 1
        return 0
    else
        return (1 - abs(t))
    end
end

function calckernel(::QuarticKernel, t)
    if abs(t) > 0
        return 0
    else
        return (15 * (1 - t^2)^2) / 16
    end
end

function calckernel(::TriweightKernel, t)
    if abs(t) > 0
        return 0
    else
        return (35 * (1 - t^2)^3) / 32
    end
end

function calckernel(::TricubeKernel, t)
    if abs(t) > 0
        return 0
    else
        return (70 * (1 - abs(t)^3)^3) / 80
    end
end

function calckernel(::GaussianKernel, t)
    exp(-0.5 * t^2) / (sqrt(2 * π))
end

function calckernel(::CosineKernel, t)
    if abs(t) > 0
        return 0
    else
        return (π * cos(π * t / 2)) / 4
    end
end

function calckernel(::LogisticKernel, t)
    1 / (exp(t) + 2 + exp(-t))
end

function calckernel(::SigmoidKernel, t)
    2 / (π * (exp(t) + exp(-t)))
end

function calckernel(::SilvermanKernel, t)
    sin(abs(t) / 2 + π / 4) * 0.5 * exp(-abs(t) / sqrt(2))
end

function construct_t1(t, tpoints)
    hcat(ones(eltype(tpoints), length(tpoints)), tpoints .- t)
end

function construct_t2(t, tpoints)
    hcat(ones(eltype(tpoints), length(tpoints)), tpoints .- t, (tpoints .- t) .^ 2)
end

function construct_w(t, tpoints, h, kernel)
    W = @. calckernel((kernel,), (tpoints - t) / h) / h
    Diagonal(W)
end

"""
```julia
u′,u = collocate_data(data,tpoints,kernel=SigmoidKernel())
u′,u = collocate_data(data,tpoints,tpoints_sample,interp,args...)
```

Computes a non-parametrically smoothed estimate of `u'` and `u`
given the `data`, where each column is a snapshot of the timeseries at
`tpoints[i]`.

For kernels, the following exist:

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
[DataInterpolations.jl](https://github.com/PumasAI/DataInterpolations.jl) to generate
data from intermediate timesteps. In this case, pass any of the methods like
`QuadraticInterpolation` as `interp`, and the timestamps to sample from as `tpoints_sample`.
"""
function collocate_data(data, tpoints, kernel = TriangularKernel())
    _one = oneunit(first(data))
    _zero = zero(first(data))
    e1 = [_one; _zero]
    e2 = [_zero; _one; _zero]
    n = length(tpoints)
    h = (n^(-1 / 5)) * (n^(-3 / 35)) * ((log(n))^(-1 / 16))

    Wd = similar(data, n, size(data, 1))
    WT1 = similar(data, n, 2)
    WT2 = similar(data, n, 3)
    x = map(tpoints) do _t
        T1 = construct_t1(_t, tpoints)
        T2 = construct_t2(_t, tpoints)
        W = construct_w(_t, tpoints, h, kernel)
        mul!(Wd, W, data')
        mul!(WT1, W, T1)
        mul!(WT2, W, T2)
        (e2' * ((T2' * WT2) \ T2')) * Wd, (e1' * ((T1' * WT1) \ T1')) * Wd
    end
    estimated_derivative = reduce(hcat, transpose.(first.(x)))
    estimated_solution = reduce(hcat, transpose.(last.(x)))
    estimated_derivative, estimated_solution
end

function collocate_data(data::AbstractVector, tpoints::AbstractVector,
                        tpoints_sample::AbstractVector,
                        interp, args...)
    u, du = collocate_data(reshape(data, 1, :), tpoints, tpoints_sample, interp, args...)
    return du[1, :], u[1, :]
end

function collocate_data(data::AbstractMatrix{T}, tpoints::AbstractVector{T},
                        tpoints_sample::AbstractVector{T}, interp, args...) where {T}
    u = zeros(T, size(data, 1), length(tpoints_sample))
    du = zeros(T, size(data, 1), length(tpoints_sample))
    for d1 in 1:size(data, 1)
        interpolation = interp(data[d1, :], tpoints, args...)
        u[d1, :] .= interpolation.(tpoints_sample)
        du[d1, :] .= DataInterpolations.derivative.((interpolation,), tpoints_sample)
    end
    return du, u
end
