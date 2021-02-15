function sciml_train(loss, θ, opt, adtype::DiffEqBase.AbstractADType = GalacticOptim.AutoZygote(), args...; kwargs...)
  optf = GalacticOptim.OptimizationFunction((x, p) -> loss(x), adtype)
  optfunc = GalacticOptim.instantiate_function(optf, θ, adtype, nothing)
  optprob = GalacticOptim.OptimizationProblem(optfunc, θ; kwargs...)
  GalacticOptim.solve(optprob, opt, args...; kwargs...)
end
