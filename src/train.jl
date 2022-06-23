function sciml_train(
    loss,
    θ,
    opt = OptimizationPolyalgorithms.PolyOpt(),
    adtype = nothing,
    args...;
    lower_bounds = nothing,
    upper_bounds = nothing,
    cb = nothing,
    callback = (args...) -> (false),
    maxiters = nothing,
    kwargs...,
)

    @warn "sciml_train is being deprecated in favor of direct usage of Optimization.jl. Please consult the Optimization.jl documentation for more details. Optimization.jl's PolyOpt solver is the polyalgorithm of sciml_train"

    if adtype === nothing
        if length(θ) < 50
            fdtime = try
                ForwardDiff.gradient(x -> first(loss(x)), θ)
                @elapsed ForwardDiff.gradient(x -> first(loss(x)), θ)
            catch
                Inf
            end
            zytime = try
                Zygote.gradient(x -> first(loss(x)), θ)
                @elapsed Zygote.gradient(x -> first(loss(x)), θ)
            catch
                Inf
            end

            if fdtime == zytime == Inf
                @warn "AD methods failed, using numerical differentiation. To debug, try ForwardDiff.gradient(loss, θ) or Zygote.gradient(loss, θ)"
                adtype = Optimization.AutoFiniteDiff()
            elseif fdtime < zytime
                adtype = Optimization.AutoForwardDiff()
            else
                adtype = Optimization.AutoZygote()
            end

        else
            adtype = Optimization.AutoZygote()
        end
    end
    if !isnothing(cb)
        callback = cb
    end

    optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
    optprob = Optimization.OptimizationProblem(
        optf,
        θ;
        lb = lower_bounds,
        ub = upper_bounds,
        kwargs...,
    )
    if maxiters !== nothing
        Optimization.solve(optprob, opt, args...; maxiters, callback = callback, kwargs...)
    else
        Optimization.solve(optprob, opt, args...; callback = callback, kwargs...)
    end
end
