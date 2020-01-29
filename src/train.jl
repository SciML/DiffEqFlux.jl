function cb(opt_state:: Optim.OptimizationState)
    cur_pred = collect(predict_adjoint(opt_state.metadata["x"]))
    n = size(training_data, 1)
    pl = scatter(1:n,training_data[:,10],label="data", legend =:bottomright,title="Spatial Plot at t=$(saveat[10])")
    scatter!(pl,1:n,cur_pred[:,10],label="prediction")
    pl2 = scatter(saveat,training_data[N÷2,:],label="data", legend =:bottomright, title="Timeseries Plot at Middle X")
    scatter!(pl2,saveat,cur_pred[N÷2,:],label="prediction")
    display(plot(pl, pl2, size=(600, 300)))
    display(opt_state.value)
    false
end

cb(trace::Optim.OptimizationTrace) = cb(last(trace))

function sciml_train!(loss_f, params, data, opt::Optim.AbstractOptimizer;cb = cb)
    result =  optimize(loss_f, params, opt, Optim.Options(extended_trace=true,callback = cb))
    return result
end

function sciml_train!(loss_f, params, data, opt::Union{Flux.Descent, Flux.Momentum, Flux.Nesterov, Flux.RMSProp, Flux.ADAM, Flux.RADAM, Flux.AdaMax, Flux.ADAGrad, Flux.ADADelta, Flux.AMSGrad, Flux.NADAM, Flux.ADAMW, Flux.Optimiser, Flux.InvDecay, Flux.ExpDecay, Flux.WeightDecay};cb = ())
    result =  Flux.train!(loss_f, params, data, opt;cb = cb)
    return result
end