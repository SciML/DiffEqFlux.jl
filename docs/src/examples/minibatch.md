# Training a Neural Ordinary Differential Equation with Mini-Batching

```julia
using DifferentialEquations, Flux, Optim, DiffEqFlux, Plots
using IterTools: ncycle 


function newtons_cooling(du, u, p, t)
    temp = u[1]
    k, temp_m = p
    du[1] = dT = -k*(temp-temp_m) 
  end

function true_sol(du, u, p, t)
    true_p = [log(2)/8.0, 100.0]
    newtons_cooling(du, u, true_p, t)
end


ann = FastChain(FastDense(1,8,tanh), FastDense(8,1,tanh))
θ = initial_params(ann)

function dudt_(u,p,t)           
    ann(u, p).* u
end

function predict_adjoint(time_batch)
    _prob = remake(prob,u0=u0,p=θ)
    Array(solve(_prob, Tsit5(), saveat = time_batch)) 
end

function loss_adjoint(batch, time_batch)
    pred = predict_adjoint(time_batch)
    sum(abs2, batch - pred)#, pred
end


u0 = Float32[200.0]
datasize = 30
tspan = (0.0f0, 3.0f0)

t = range(tspan[1], tspan[2], length=datasize)
true_prob = ODEProblem(true_sol, u0, tspan)
ode_data = Array(solve(true_prob, Tsit5(), saveat=t))

prob = ODEProblem{false}(dudt_, u0, tspan, θ)

k = 10
train_loader = Flux.Data.DataLoader(ode_data, t, batchsize = k)

for (x, y) in train_loader
    @show x
    @show y
end

numEpochs = 300
losses=[]
cb() = begin
    l=loss_adjoint(ode_data, t)
    push!(losses, l)
    @show l
    pred=predict_adjoint(t)
    pl = scatter(t,ode_data[1,:],label="data", color=:black, ylim=(150,200))
    scatter!(pl,t,pred[1,:],label="prediction", color=:darkgreen)
    display(plot(pl))
    false
end 

opt=ADAM(0.05)
Flux.train!(loss_adjoint, Flux.params(θ), ncycle(train_loader,numEpochs), opt, cb=Flux.throttle(cb, 10))

#Now lets see how well it generalizes to new initial conditions 

starting_temp=collect(10:30:250)
true_prob_func(u0)=ODEProblem(true_sol, [u0], tspan)
color_cycle=palette(:tab10)
pl=plot()
for (j,temp) in enumerate(starting_temp)
    ode_test_sol = solve(ODEProblem(true_sol, [temp], (0.0f0,10.0f0)), Tsit5(), saveat=0.0:0.5:10.0)
    ode_nn_sol = solve(ODEProblem{false}(dudt_, [temp], (0.0f0,10.0f0), θ))
    scatter!(pl, ode_test_sol, var=(0,1), label="", color=color_cycle[j])
    plot!(pl, ode_nn_sol, var=(0,1), label="", color=color_cycle[j], lw=2.0)
end
display(pl) 
title!("Neural ODE for Newton's Law of Cooling: Test Data")
xlabel!("Time")
ylabel!("Temp") 


# How to use MLDataUtils 
using MLDataUtils
train_loader, _, _ = kfolds((ode_data, t))

@info "Now training using the MLDataUtils format"
Flux.train!(loss_adjoint, Flux.params(θ), ncycle(eachbatch(train_loader[1], k), numEpochs), opt, cb=Flux.throttle(cb, 10))
```

When training a neural network we need to find the gradient with respect to our data set. There are three main ways to partition our data when using a training algorithm like gradient descent: stochastic, batching and mini-batching. Stochastic gradient descent trains on a single random data point each epoch. This allows for the neural network to better converge to the global minimum even on noisy data but is computationally inefficient. Batch gradient descent trains on the whole data set each epoch and while computationally effiecient is prone to converging to local minima. Mini-batching combines both of these advantages and by training on a small random "mini-batch" of the data each epoch can converge to the global minimum while remaining more computationally effiecient than stochastic descent. Typically we do this by randomly selecting subsets of the data each epoch and use this subset to train on. We can also pre-batch the data by creating an iterator holding these randomly selected batches before beginning to train. The proper size for the batch can be determined expirementally. Let us see how to do this with Julia. 

For this example we will use a very simple ordinary differential equation, newtons law of cooling. We can represent this in Julia like so. 

```julia
using DifferentialEquations, Flux, Optim, DiffEqFlux, Plots
using IterTools: ncycle 


function newtons_cooling(du, u, p, t)
    temp = u[1]
    k, temp_m = p
    du[1] = dT = -k*(temp-temp_m) 
  end

function true_sol(du, u, p, t)
    true_p = [log(2)/8.0, 100.0]
    newtons_cooling(du, u, true_p, t)
end
```

Now we define a neural-network using a linear approximation with 1 hidden layer of 8 neurons.  

```julia
ann = FastChain(FastDense(1,8,tanh), FastDense(8,1,tanh))
θ = initial_params(ann)

function dudt_(u,p,t)           
    ann(u, p).* u
end
```

From here we build a loss function around it. 

```julia
function predict_adjoint(time_batch)
    Array(concrete_solve(prob, Tsit5(),
    u0, θ, saveat = time_batch)) 
end

function loss_adjoint(batch, time_batch)
    pred = predict_adjoint(time_batch)
    sum(abs2, batch - pred)#, pred
end
```

To add support for batches of size `k` we use `Flux.Data.DataLoader`. To use this we pass in the `ode_data` and `t` as the 'x' and 'y' data to batch respectively. The parameter `batchsize` controls the size of our batches. We check our implementation by iterating over the batched data. 

```julia
u0 = Float32[200.0]
datasize = 30
tspan = (0.0f0, 3.0f0)

t = range(tspan[1], tspan[2], length=datasize)
true_prob = ODEProblem(true_sol, u0, tspan)
ode_data = Array(solve(true_prob, Tsit5(), saveat=t))
prob = ODEProblem{false}(dudt_, u0, tspan, θ)

k = 10
train_loader = Flux.Data.DataLoader(ode_data, t, batchsize = k)
for (x, y) in train_loader
    @show x
    @show y
end


#x = Float32[200.0 199.55284 199.1077 198.66454 198.22334 197.78413 197.3469 196.9116 196.47826 196.04686]
#y = Float32[0.0, 0.05172414, 0.10344828, 0.15517241, 0.20689656, 0.25862068, 0.31034482, 0.36206895, 0.41379312, 0.46551725]
#x = Float32[195.61739 195.18983 194.76418 194.34044 193.9186 193.49864 193.08057 192.66435 192.25 191.8375]
#y = Float32[0.51724136, 0.5689655, 0.62068963, 0.67241377, 0.7241379, 0.7758621, 0.82758623, 0.87931037, 0.9310345, 0.98275864]
#x = Float32[191.42683 191.01802 190.61102 190.20586 189.8025 189.40094 189.00119 188.60321 188.20702 187.8126]
#y = Float32[1.0344827, 1.0862069, 1.137931, 1.1896552, 1.2413793, 1.2931035, 1.3448275, 1.3965517, 1.4482758, 1.5]
```

Now we train the neural network with a user defined call back function to display loss and the graphs with a maximum of 300 epochs. 

```julia
numEpochs = 300
losses=[]
cb() = begin
    l=loss_adjoint(ode_data, t)
    push!(losses, l)
    @show l
    pred=predict_adjoint(t)
    pl = scatter(t,ode_data[1,:],label="data", color=:black, ylim=(150,200))
    scatter!(pl,t,pred[1,:],label="prediction", color=:darkgreen)
    display(plot(pl))
    false
end 

opt=ADAM(0.05)
Flux.train!(loss_adjoint, Flux.params(θ), ncycle(train_loader,numEpochs), opt, cb=Flux.throttle(cb, 10))
```

Finally we can see how well our trained network will generalize to new initial conditions. 

```julia
starting_temp=collect(10:30:250)
true_prob_func(u0)=ODEProblem(true_sol, [u0], tspan)
color_cycle=palette(:tab10)
pl=plot()
for (j,temp) in enumerate(starting_temp)
    ode_test_sol = solve(ODEProblem(true_sol, [temp], (0.0f0,10.0f0)), Tsit5(), saveat=0.0:0.5:10.0)
    ode_nn_sol = solve(ODEProblem{false}(dudt_, [temp], (0.0f0,10.0f0), θ))
    scatter!(pl, ode_test_sol, var=(0,1), label="", color=color_cycle[j])
    plot!(pl, ode_nn_sol, var=(0,1), label="", color=color_cycle[j], lw=2.0)
end
display(pl) 
title!("Neural ODE for Newton's Law of Cooling: Test Data")
xlabel!("Time")
ylabel!("Temp")
```

We can also minibatch using tools from `MLDataUtils`. To do this we need to slightly change our implementation and is shown below again with a batch size of k and the same number of epochs.

```julia
using MLDataUtils
train_loader, _, _ = kfolds((ode_data, t))

@info "Now training using the MLDataUtils format"
Flux.train!(loss_adjoint, Flux.params(θ), ncycle(eachbatch(train_loader[1], k), numEpochs), opt, cb=Flux.throttle(cb, 10))
```
