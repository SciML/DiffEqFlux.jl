using DiffEqFlux, Optimization, OrdinaryDiffEq, RecursiveArrayTools


u0 = Float32[0.0; 2.0]
du0 = Float32[0.0; 0.0]
tspan = (0.0f0, 1.0f0)
t = range(tspan[1], tspan[2], length = 20)

model = FastChain(FastDense(2, 50, tanh), FastDense(50, 2))
p = initial_params(model)
ff(du, u, p, t) = model(u, p)
prob = SecondOrderODEProblem{false}(ff, du0, u0, tspan, p)

function predict(p)
    Array(
        solve(
            prob,
            Tsit5(),
            p = p,
            saveat = t,
            sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP()),
        ),
    )
end

correct_pos = Float32.(transpose(hcat(collect(0:0.05:1)[2:end], collect(2:-0.05:1)[2:end])))

function loss_n_ode(p)
    pred = predict(p)
    sum(abs2, correct_pos .- pred[1:2, :]), pred
end

data = Iterators.repeated((), 1000)
opt = ADAM(0.01)

l1 = loss_n_ode(p)

callback = function (p, l, pred)
    @show l
    l < 0.01 && Flux.stop()
end

optfunc =
    Optimization.OptimizationFunction((x, p) -> loss_n_ode(x), Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optfunc, p)
res = Optimization.solve(optprob, opt, callback = callback, maxiters = 100)
l2 = loss_n_ode(res.minimizer)
@test l2 < l1

function predict(p)
    Array(
        solve(
            prob,
            Tsit5(),
            p = p,
            saveat = t,
            sensealg = QuadratureAdjoint(autojacvec = ZygoteVJP()),
        ),
    )
end

correct_pos = Float32.(transpose(hcat(collect(0:0.05:1)[2:end], collect(2:-0.05:1)[2:end])))

function loss_n_ode(p)
    pred = predict(p)
    sum(abs2, correct_pos .- pred[1:2, :]), pred
end

data = Iterators.repeated((), 1000)
opt = ADAM(0.01)

loss_n_ode(p)

callback = function (p, l, pred)
    @show l
    l < 0.01 && Flux.stop()
end
optfunc =
    Optimization.OptimizationFunction((x, p) -> loss_n_ode(x), Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optfunc, p)
res = Optimization.solve(optprob, opt, callback = callback, maxiters = 100)
l2 = loss_n_ode(res.minimizer)
@test l2 < l1

function predict(p)
    Array(
        solve(
            prob,
            Tsit5(),
            p = p,
            saveat = t,
            sensealg = BacksolveAdjoint(autojacvec = ZygoteVJP()),
        ),
    )
end

correct_pos = Float32.(transpose(hcat(collect(0:0.05:1)[2:end], collect(2:-0.05:1)[2:end])))

function loss_n_ode(p)
    pred = predict(p)
    sum(abs2, correct_pos .- pred[1:2, :]), pred
end

data = Iterators.repeated((), 1000)
opt = ADAM(0.01)

loss_n_ode(p)

callback = function (p, l, pred)
    @show l
    l < 0.01 && Flux.stop()
end

optfunc =
    Optimization.OptimizationFunction((x, p) -> loss_n_ode(x), Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optfunc, p)
res = Optimization.solve(optprob, opt, callback = callback, maxiters = 100)
l2 = loss_n_ode(res.minimizer)
@test l2 < l1
