using DiffEqFlux, Flux, Plots, Statistics
using Random
Random.seed!(1234); # Fix seed

N = 48
π_32 = Float32(π)
t = range(0.0f0,stop=3π_32, length = N)
sin_t = sin.(t)
cos_t = cos.(t)
data_x = vcat(reshape(sin_t,(1,N)), reshape(cos_t,(1,N)))
data_y = reshape(sin.(range(0.0f0,stop=6π_32, length = N)), (1, N))
data_x = [data_x[:,i] for i=1:N]
data_y = [[data_y[i]] for i=1:N]

println(size(data_x))
println(size(data_y))


m = Chain(LTC(2,32), Dense(32,1,x->x))
function loss_(x,y)
    diff = (m.(x) .- y)
    diff = [diff[i][1] for i=1:N]
    mean(abs2.(diff))
end

#callback function to observe training
cb = function ()
  cur_pred = m.(data_x)
  pl = plot(t, [data_y[i][1] for i=1:length(data_y)], label="data")
  plot!(pl, t, [cur_pred[i][1] for i=1:length(cur_pred)], label="prediction")
  display(plot(pl))
  @show loss_(data_x, data_y)
end

ps = Flux.params(m);

opt = Flux.ADAM(0.05)
epochs = 400
for epoch in 1:epochs
        x, y = data_x[:,1], data_y[:,1]
        gs = Flux.gradient(ps) do
            loss_(x, y)
        end
        Flux.Optimise.update!(opt, ps, gs)
        Flux.reset!(m)
        if epoch % 10 == 0
            @show epoch
            cb()
        end
end
