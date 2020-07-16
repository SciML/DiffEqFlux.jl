# Training a Neural Ordinary Differntial Equation with Mini-Batching

```julia
using DifferentialEquations, Flux, Optim, DiffEqFlux, Plots

function newtons_cooling(du, u, p, t)
    temp = u[1]
    k, temp_m = p
    du[1] = dT = -k*(temp-temp_m) 
  end

function true_sol(du, u, p, t)
    true_p = [log(2)/8.0, 100.0]
    newtons_cooling(du, u, true_p, t)
end
         
function dudt_(u,p,t)           
    ann(u, p).* u
end

function predict_adjoint(fullp, time_batch)
    Array(concrete_solve(prob, Tsit5(),
    u0, fullp, saveat = time_batch)) 
end

function loss_adjoint(fullp, batch, time_batch)
    pred = predict_adjoint(fullp,time_batch)
    sum(abs2, batch - pred), pred
end

cb = function (p,l,pred;doplot=false) #callback function to observe training
    display(l)
    # plot current prediction against data
    if doplot
      pl = scatter(t,ode_data[1,:],label="data")
      scatter!(pl,t,pred[1,:],label="prediction")
      display(plot(pl))
    end
    return false
end

u0 = Float32[200.0]
datasize = 30
tspan = (0.0f0, 1.5f0)

t = range(tspan[1], tspan[2], length=datasize)
true_prob = ODEProblem(true_sol, u0, tspan)
ode_data = Array(solve(true_prob, Tsit5(), saveat=t))

ann = FastChain(FastDense(1,8,tanh), FastDense(8,1,tanh))
pp = initial_params(ann)
prob = ODEProblem{false}(dudt_, u0, tspan, pp)


k = 10
train_loader = Flux.Data.DataLoader(ode_data, t, batchsize = k)

numEpochs = 300

using IterTools: ncycle
res1 = DiffEqFlux.sciml_train(loss_adjoint, pp, ADAM(0.05), ncycle(train_loader, numEpochs), cb = cb, maxiters = numEpochs)
cb(res1.minimizer,loss_adjoint(res1.minimizer, ode_data, t)...;doplot=true)

```


When training a neural network we need to find the gradient with respect to our data set. There are three main ways to partition our data when using a training algorithm like gradient descent: stochastic, batching and mini-batching. Stochastic gradient descent trains on a single random data point each epoch. This allows for the neural network to better converge to the global minimum even on noisy data but is computationally inefficient. Batch gradient descent trains on the whole data set each epoch and while computationally effiecient is prone to converging to local minima. Mini-batching combines both of these advantages and by training on a small random "mini-batch" of the data each epoch can converge to the global minimum while remaining more computationally effiecient than stochastic descent. Typically we do this by randomly selecting subsets of the data each epoch and use this subset to train on. We can also pre-batch the data by creating an iterator holding these randomly selected batches before beginning to train. The proper size for the batch can be determined expirementally. Let us see how to do this with Julia. 




For this example we will use a very simple ordinary differential equation, newtons law of cooling. We can represent this in Julia like so. 



```julia
using DifferentialEquations, Flux, Optim, DiffEqFlux, Plots

function newtons_cooling(du, u, p, t)
    temp = u[1]
    k, temp_m = p
    du[1] = dT = -k*(temp-temp_m) 
  end

function true_sol(du, u, p, t)
    true_p = [log(2)/8.0, 100.0]
    newtons_cooling(du, u, true_p, t)
end

u0 = Float32[200.0]
datasize = 30
tspan = (0.0f0, 1.5f0)

t = range(tspan[1], tspan[2], length=datasize)
true_prob = ODEProblem(true_sol, u0, tspan)
ode_data = Array(solve(true_prob, Tsit5(), saveat=t))

```

Now we define a neural-network using a linear approximation with 1 hidden layer of 8 neurons.  

```julia
ann = FastChain(FastDense(1,8,tanh), FastDense(8,1,tanh))
pp = initial_params(ann)
prob = ODEProblem{false}(dudt_, u0, tspan, pp)

function dudt_(u,p,t)           
    ann(u, p).* u
end
```


From here we build a loss function around it. 

```julia
function predict_adjoint(fullp, time_batch)
    Array(concrete_solve(prob, Tsit5(),
    u0, fullp, saveat = time_batch)) 
end

function loss_adjoint(fullp, batch, time_batch)
    pred = predict_adjoint(fullp,time_batch)
    sum(abs2, batch - pred), pred
end
```

To add support for batches of size `k` we use `Flux.Data.DataLoader`. To use this we pass in the `ode_data` and `t` as the 'x' and 'y' data to batch respectively. The parameter `batchsize` controls the size of our batches. We check our implementation by iterating over the batched data. 

```julia
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
cb = function (p,l,pred;doplot=false) #callback function to observe training
    display(l)
    # plot current prediction against data
    if doplot
      pl = scatter(t,ode_data[1,:],label="data")
      scatter!(pl,t,pred[1,:],label="prediction")
      display(plot(pl))
    end
    return false
end

using IterTools: ncycle
res1 = DiffEqFlux.sciml_train(loss_adjoint, pp, ADAM(0.05), ncycle(train_loader, numEpochs), cb = cb, maxiters = numEpochs)
cb(res1.minimizer,loss_adjoint(res1.minimizer, ode_data, t)...;doplot=true)
```

We can also minibatch using tools from `MLDataUtils`. To do this we need to slightly change our implementation and is shown below again with a batch size of k and the same number of epochs.

```julia
using MLDataUtils
train_loader, _, _ = kfolds((ode_data, t))

res1 = DiffEqFlux.sciml_train(loss_adjoint, pp, ADAM(0.05), ncycle(eachbatch(train_loader[1], k), numEpochs), cb = cb, maxiters = numEpochs)
cb(res1.minimizer,loss_adjoint(res1.minimizer, ode_data, t)...;doplot=true)

```



