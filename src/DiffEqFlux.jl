module DiffEqFlux

using DiffEqBase, Flux, DiffResults, DiffEqSensitivity, ForwardDiff

include("Flux/layers.jl")
include("Flux/neural_ode.jl")
include("Flux/utils.jl")

export diffeq_fd, diffeq_rd, diffeq_adjoint, neural_ode
end
