abstract type AbstractSplineLayer <: Function end
Flux.trainable(m::AbstractSplineLayer) = (m.p,)

struct SplineLayer{T<:Tuple{Real, Real},R<:Real,S1<:AbstractVector,S2<:UnionAll} <: AbstractSplineLayer
    time_span::T
    time_step::R
    saved_points::S1
    spline_basis::S2
    function SplineLayer(time_span,time_step,spline_basis,saved_points=nothing)
        saved_points = randn(length(time_span[1]:time_step:time_span[2]))
        new{typeof(time_span),typeof(time_step),typeof(saved_points),typeof(spline_basis)}(time_span,time_step,saved_points,spline_basis)
    end
end

function (layer::SplineLayer)(t::Real,p=layer.saved_points)
    return layer.spline_basis(p,layer.time_span[1]:layer.time_step:layer.time_span[2])
end
