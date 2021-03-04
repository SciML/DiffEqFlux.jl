function sciml_train(loss, θ, opt, adtype::DiffEqBase.AbstractADType = GalacticOptim.AutoZygote(), args...;
                     lower_bounds = nothing, upper_bounds = nothing, kwargs...)
  optf = GalacticOptim.OptimizationFunction((x, p) -> loss(x), adtype)
  optfunc = GalacticOptim.instantiate_function(optf, θ, adtype, nothing)
  optprob = GalacticOptim.OptimizationProblem(optfunc, θ; lb = lower_bounds, ub = upper_bounds, kwargs...)
  GalacticOptim.solve(optprob, opt, args...; kwargs...)
end
