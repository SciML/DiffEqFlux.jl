module DiffEqFlux

using DiffEqBase, Flux, DiffResults, DiffEqSensitivity, ForwardDiff,
      Requires, RecursiveArrayTools, Adapt, LinearAlgebra

gpu_or_cpu(x) = Array
function __init__()
    @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
        gpu_or_cpu(x::CuArrays.CuArray) = CuArrays.CuArray
        gpu_or_cpu(x::TrackedArray{<:Any,<:Any,<:CuArrays.CuArray}) = CuArrays.CuArray
        gpu_or_cpu(x::Transpose{<:Any,<:CuArrays.CuArray}) = CuArrays.CuArray
        gpu_or_cpu(x::Adjoint{<:Any,<:CuArrays.CuArray}) = CuArrays.CuArray
        gpu_or_cpu(x::Adjoint{<:Any,TrackedArray{<:Any,<:Any,<:CuArrays.CuArray}}) = CuArrays.CuArray
        gpu_or_cpu(x::Transpose{<:Any,TrackedArray{<:Any,<:Any,<:CuArrays.CuArray}}) = CuArrays.CuArray
    end
end

include("Flux/layers.jl")
include("Flux/neural_de.jl")
include("Flux/utils.jl")

export diffeq_fd, diffeq_rd, diffeq_adjoint
export neural_ode, neural_ode_rd
export neural_dmsde
end
