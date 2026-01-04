module DiffEqFluxDataInterpolationsExt

using DataInterpolations: DataInterpolations
using DiffEqFlux: DiffEqFlux

@views function DiffEqFlux.collocate_data(
        data::AbstractMatrix{T}, tpoints::AbstractVector{T},
        tpoints_sample::AbstractVector{T}, interp, args...
    ) where {T}
    u = zeros(T, size(data, 1), length(tpoints_sample))
    du = zeros(T, size(data, 1), length(tpoints_sample))
    for d1 in axes(data, 1)
        interpolation = interp(data[d1, :], tpoints, args...)
        u[d1, :] .= interpolation.(tpoints_sample)
        du[d1, :] .= DataInterpolations.derivative.((interpolation,), tpoints_sample)
    end
    return du, u
end

end
