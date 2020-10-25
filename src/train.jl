function sciml_train(loss, θ, opt, adtype::DiffEqBase.AbstractADType = GalacticOptim.AutoZygote(), args...; kwargs...)
  @warn("sciml_train has been deprecated in favor of GalacticOptim.jl (https://github.com/SciML/GalacticOptim.jl)") 
  optfunc = GalacticOptim.instantiate_function((x, p) -> loss(x), θ, adtype, nothing)
  optprob = GalacticOptim.OptimizationProblem(optfunc, θ; kwargs...)
  GalacticOptim.solve(optprob, opt, args...; kwargs...)
end
