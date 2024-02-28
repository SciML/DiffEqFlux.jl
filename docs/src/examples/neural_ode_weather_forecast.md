# Weather forecasting with neural ODEs

In this example we are going to apply neural ODEs to a multidimensional weather dataset and use it for weather forecasting.
This example is adapted from [Forecasting the weather with neural ODEs - Sebatian Callh personal blog](https://sebastiancallh.github.io/post/neural-ode-weather-forecast/).

## The data

The data is a four-dimensional dataset of daily temperature, humidity, wind speed and pressure measured over four years in the city Delhi. Let us download and plot it.

```@example weather_forecast
using Random, Dates, Optimization, ComponentArrays, Lux, OptimizationOptimisers, DiffEqFlux,
      OrdinaryDiffEq, CSV, DataFrames, Dates, Statistics, Plots, DataDeps

function download_data(
        data_url = "https://raw.githubusercontent.com/SebastianCallh/neural-ode-weather-forecast/master/data/",
        data_local_path = "./delhi")
    function load(file_name)
        data_dep = DataDep("delhi/train", "", "$data_url/$file_name")
        Base.download(data_dep, data_local_path; i_accept_the_terms_of_use = true)
        CSV.read(joinpath(data_local_path, file_name), DataFrame)
    end

    train_df = load("DailyDelhiClimateTrain.csv")
    test_df = load("DailyDelhiClimateTest.csv")
    return vcat(train_df, test_df)
end

df = download_data()
first(df, 5) # hide
```

```@example weather_forecast
FEATURES = [:meantemp, :humidity, :wind_speed, :meanpressure]
UNITS = ["Celsius", "g/m³ of water", "km/h", "hPa"]
FEATURE_NAMES = ["Mean temperature", "Humidity", "Wind speed", "Mean pressure"]

function plot_data(df)
    plots = map(enumerate(zip(FEATURES, FEATURE_NAMES, UNITS))) do (i, (f, n, u))
        plot(df[:, :date], df[:, f]; title = n, label = nothing,
            ylabel = u, size = (800, 600), color = i)
    end

    n = length(plots)
    plot(plots...; layout = (Int(n / 2), Int(n / 2)))
end

plot_data(df)
```

The data show clear annual behaviour (it is difficult to see for pressure due to wild measurement errors but the pattern is there).
It is concievable that this system can be described with an ODE, but which? Let us use an network to learn the dynamics from the dataset.
Training neural networks is easier with standardised data so we will compute standardised features before training. Finally, we take the first 20 days for training and the rest for testing.

```@example weather_forecast
function standardize(x)
    μ = mean(x; dims = 2)
    σ = std(x; dims = 2)
    z = (x .- μ) ./ σ
    return z, μ, σ
end

function featurize(raw_df, num_train = 20)
    raw_df.year = Float64.(year.(raw_df.date))
    raw_df.month = Float64.(month.(raw_df.date))
    df = combine(groupby(raw_df, [:year, :month]),
        :date => (d -> mean(year.(d)) .+ mean(month.(d)) ./ 12),
        :meantemp => mean,
        :humidity => mean,
        :wind_speed => mean,
        :meanpressure => mean;
        renamecols = false)
    t_and_y(df) = df.date', Matrix(select(df, FEATURES))'
    t_train, y_train = t_and_y(df[1:num_train, :])
    t_test, y_test = t_and_y(df[(num_train + 1):end, :])
    t_train, t_mean, t_scale = standardize(t_train)
    y_train, y_mean, y_scale = standardize(y_train)
    t_test = (t_test .- t_mean) ./ t_scale
    y_test = (y_test .- y_mean) ./ y_scale

    return (vec(t_train), y_train,
        vec(t_test), y_test,
        (t_mean, t_scale),
        (y_mean, y_scale))
end

function plot_features(t_train, y_train, t_test, y_test)
    plt_split = plot(reshape(t_train, :), y_train';
        linewidth = 3, colors = 1:4,
        xlabel = "Normalized time",
        ylabel = "Normalized values",
        label = nothing,
        title = "Features")
    plot!(plt_split, reshape(t_test, :), y_test';
        linewidth = 3, linestyle = :dash,
        color = [1 2 3 4], label = nothing)
    plot!(plt_split, [0], [0]; linewidth = 0,
        label = "Train", color = 1)
    plot!(plt_split, [0], [0]; linewidth = 0,
        linestyle = :dash, label = "Test",
        color = 1,
        ylims = (-5, 5))
end

t_train, y_train, t_test, y_test, (t_mean, t_scale), (y_mean, y_scale) = featurize(df)
plot_features(t_train, y_train, t_test, y_test)
```

The dataset is now centered around 0 with a standard deviation of 1.
We will ignore the extreme pressure measurements for simplicity.
Since they are in the test split they won't impact training anyway.
We are now ready to construct and train our model! To avoid local minimas we will train iteratively with increasing amounts of data.

```@example weather_forecast
function neural_ode(t, data_dim)
    f = Chain(Dense(data_dim => 64, swish), Dense(64 => 32, swish), Dense(32 => data_dim))

    node = NeuralODE(f, extrema(t), Tsit5(); saveat = t,
        abstol = 1e-9, reltol = 1e-9)

    rng = Random.default_rng()
    p, state = Lux.setup(rng, f)

    return node, ComponentArray(p), state
end

function train_one_round(node, p, state, y, opt, maxiters, rng, y0 = y[:, 1]; kwargs...)
    predict(p) = Array(node(y0, p, state)[1])
    loss(p) = sum(abs2, predict(p) .- y)

    adtype = Optimization.AutoZygote()
    optf = OptimizationFunction((p, _) -> loss(p), adtype)
    optprob = OptimizationProblem(optf, p)
    res = solve(optprob, opt; maxiters = maxiters, kwargs...)
    res.minimizer, state
end

function train(t, y, obs_grid, maxiters, lr, rng, p = nothing, state = nothing; kwargs...)
    log_results(ps, losses) = (p, loss) -> begin
        push!(ps, copy(p.u))
        push!(losses, loss)
        false
    end

    ps, losses = ComponentArray[], Float32[]
    for k in obs_grid
        node, p_new, state_new = neural_ode(t, size(y, 1))
        p === nothing && (p = p_new)
        state === nothing && (state = state_new)

        p, state = train_one_round(
            node, p, state, y, OptimizationOptimisers.AdamW(lr), maxiters, rng;
            callback = log_results(ps, losses), kwargs...)
    end
    ps, state, losses
end

rng = MersenneTwister(123)
obs_grid = 4:4:length(t_train) # we train on an increasing amount of the first k obs
maxiters = 150
lr = 5e-3
ps, state, losses = train(t_train, y_train, obs_grid, maxiters, lr, rng; progress = true)
```

We can now animate the training to get a better understanding of the fit.

```@example weather_forecast
predict(y0, t, p, state) = begin
    node, _, _ = neural_ode(t, length(y0))
    Array(node(y0, p, state)[1])
end

function plot_pred(t_train, y_train, t_grid, rescale_t, rescale_y, num_iters, p, state,
        loss, y0 = y_train[:, 1])
    y_pred = predict(y0, t_grid, p, state)
    return plot_result(rescale_t(t_train), rescale_y(y_train), rescale_t(t_grid),
        rescale_y(y_pred), loss, num_iters)
end

function plot_pred(t, y, y_pred)
    plt = Plots.scatter(t, y; label = "Observation")
    Plots.plot!(plt, t, y_pred; label = "Prediction")
end

function plot_pred(t, y, t_pred, y_pred; kwargs...)
    plot_params = zip(eachrow(y), eachrow(y_pred), FEATURE_NAMES, UNITS)
    map(enumerate(plot_params)) do (i, (yᵢ, ŷᵢ, name, unit))
        plt = Plots.plot(t_pred, ŷᵢ; label = "Prediction", color = i, linewidth = 3,
            legend = nothing, title = name, kwargs...)
        Plots.scatter!(plt, t, yᵢ; label = "Observation", xlabel = "Time", ylabel = unit,
            markersize = 5, color = i)
    end
end

function plot_result(t, y, t_pred, y_pred, loss, num_iters; kwargs...)
    plts_preds = plot_pred(t, y, t_pred, y_pred; kwargs...)
    plot!(plts_preds[1]; ylim = (10, 40), legend = (0.65, 1.0))
    plot!(plts_preds[2]; ylim = (20, 100))
    plot!(plts_preds[3]; ylim = (2, 12))
    plot!(plts_preds[4]; ylim = (990, 1025))

    p_loss = Plots.plot(loss; label = nothing, linewidth = 3,
        title = "Loss", xlabel = "Iterations", xlim = (0, num_iters))
    plots = [plts_preds..., p_loss]
    plot(plots...; layout = grid(length(plots), 1), size = (900, 900))
end

function animate_training(plot_frame, t_train, y_train, ps, losses, obs_grid;
        pause_for = 300)
    obs_count = Dict(i - 1 => n for (i, n) in enumerate(obs_grid))
    is = [min(i, length(losses)) for i in 2:(length(losses) + pause_for)]
    @animate for i in is
        stage = Int(floor((i - 1) / length(losses) * length(obs_grid)))
        k = obs_count[stage]
        plot_frame(t_train[1:k], y_train[:, 1:k], ps[i], losses[1:i])
    end every 2
end

num_iters = length(losses)
t_train_grid = collect(range(extrema(t_train)...; length = 500))
rescale_t(x) = t_scale .* x .+ t_mean
rescale_y(x) = y_scale .* x .+ y_mean
function plot_frame(t, y, p, loss)
    plot_pred(t, y, t_train_grid, rescale_t, rescale_y, num_iters, p, state, loss)
end
anim = animate_training(plot_frame, t_train, y_train, ps, losses, obs_grid)
gif(anim, "node_weather_forecast_training.gif")
```

Looks good! But how well does the model forecast?

```@example weather_forecast
function plot_extrapolation(t_train, y_train, t_test, y_test, t̂, ŷ)
    plts = plot_pred(t_train, y_train, t̂, ŷ)
    for (i, (plt, y)) in enumerate(zip(plts, eachrow(y_test)))
        scatter!(plt, t_test, y; color = i, markerstrokecolor = :white,
            label = "Test observation")
    end

    plot!(plts[1]; ylim = (10, 40), legend = :topleft)
    plot!(plts[2]; ylim = (20, 100))
    plot!(plts[3]; ylim = (2, 12))
    plot!(plts[4]; ylim = (990, 1025))
    plot(plts...; layout = grid(length(plts), 1), size = (900, 900))
end

t_grid = collect(range(minimum(t_train), maximum(t_test); length = 500))
y_pred = predict(y_train[:, 1], t_grid, ps[end], state)
plot_extrapolation(rescale_t(t_train), rescale_y(y_train), rescale_t(t_test),
    rescale_y(y_test), rescale_t(t_grid), rescale_y(y_pred))
```

While there is some drift in the weather patterns, the model extrapolates very well!
