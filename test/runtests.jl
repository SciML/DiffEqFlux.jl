using DiffEqFlux, Test

@testset "DiffEqFlux" begin

include("layers.jl")
include("utils.jl")
include("neural_de.jl")
include("partial_neural.jl")

end
