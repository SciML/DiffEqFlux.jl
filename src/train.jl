function sciml_train(loss, θ, opt, adtype = nothing, args...;
                     lower_bounds = nothing, upper_bounds = nothing, kwargs...)

  if adtype === nothing
      if length(θ) < 100
          adtype = GalacticOptim.AutoForwardDiff()
      else
          adtype = GalacticOptim.AutoZygote()
      end
  end

  optf = GalacticOptim.OptimizationFunction((x, p) -> loss(x), adtype)
  optfunc = GalacticOptim.instantiate_function(optf, θ, adtype, nothing)
  optprob = GalacticOptim.OptimizationProblem(optfunc, θ; lb = lower_bounds, ub = upper_bounds, kwargs...)
  GalacticOptim.solve(optprob, opt, args...; kwargs...)
end
