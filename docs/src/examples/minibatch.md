# Training a Neural Ordinary Differntial Equation with Mini-Batching

When training universal differential equations it is often helpful to batch our data. This is particularly useful when working with large sets of training data. Let us take a look at how this works with the Lotka-Volterra equation. 

We first get a time series array from the Lotka-Volterra equation as data:



```julia
using DifferentialEquations, Flux, Optim, DiffEqFlux

u0 = Float32[2.; 0.]
datasize = 30
tspan = (0.0f0,1.5f0)

function trueODEfunc(du,u,p,t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end
t = range(tspan[1],tspan[2],length=datasize)
prob = ODEProblem(trueODEfunc,u0,tspan)
ode_data = Array(solve(prob,Tsit5(),saveat=t))
```

Now let's define a neural network with a `NeuralODE` layer. First we define the layer. Here we're going to use `FastChain`, which is a faster neural network structure for NeuralODEs:

```julia
dudt2 = FastChain((x,p) -> x.^3,
            FastDense(2,50,tanh),
            FastDense(50,2))
n_ode = NeuralODE(dudt2,tspan,Tsit5(),saveat=t)
```

In our model we used the `x -> x.^3` assumption in the model. By incorporating structure into our equations, we can reduce the required size and training time for the neural network, but a good guess needs to be known!

From here we build a loss function around it. The `NeuralODE` has an optional second argument for new parameters which we will use to iteratively change the neural network in our training loop. We will use the network's output against the time series data. To add support for batches of size `k` we need to add parameters representing the start index and size of our batch to our loss function:

```julia
function predict_n_ode(p)
    n_ode(u0,p)
  end
  
  function loss_n_ode(p, start, k)
      pred = predict_n_ode(p)
      loss = sum(abs2,ode_data[:,start:start+k] .- pred[:,start:start+k])
      loss,pred
  end
  
```

and then create a generator, that produces `MAX_BATCHES` random tuples `(start index, batch size)`:

```julia
  MAX_BATCHES = 1000
  k = 15 #batch size
  data = ((rand(1:size(ode_data)[2] -k), k) for i in 1:MAX_BATCHES)
```
and then train the neural network to learn the ODE:

```julia

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


 
res = DiffEqFlux.sciml_train(loss_n_ode, res1.minimizer, LBFGS(), data, cb = cb)

```




