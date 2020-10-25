function sciml_train(loss, θ, opt, adtype::DiffEqBase.AbstractADType = GalacticOptim.AutoZygote(); kwargs...)
  @warn("sciml_train has been deprecated in favor of GalacticOptim.jl (https://github.com/SciML/GalacticOptim.jl)") 
  optfunc = GalacticOptim.instantiate_function((x, p) -> loss(x), θ, adtype)
  optprob = GalacticOptim.OptimizationProblem(optfunc, θ; kwargs...)
  GalacticOptim.solve(optprob, opt; kwargs...)
end
