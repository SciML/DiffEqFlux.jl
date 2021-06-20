# Partial Differential Equation (PDE) Constrained Optimization

This example uses a prediction model to optimize the one-dimensional Heat Equation.
(Step-by-step description below)

```julia
using DelimitedFiles,Plots
using DifferentialEquations, DiffEqFlux

# Problem setup parameters:
Lx = 10.0
x  = 0.0:0.01:Lx
dx = x[2] - x[1]
Nx = size(x)

u0 = exp.(-(x.-3.0).^2) # I.C

## Problem Parameters
p        = [1.0,1.0]    # True solution parameters
xtrs     = [dx,Nx]      # Extra parameters
dt       = 0.40*dx^2    # CFL condition
t0, tMax = 0.0 ,1000*dt
tspan    = (t0,tMax)
t        = t0:dt:tMax;

## Definition of Auxiliary functions
function ddx(u,dx)
    """
    2nd order Central difference for 1st degree derivative
    """
    return [[zero(eltype(u))] ; (u[3:end] - u[1:end-2]) ./ (2.0*dx) ; [zero(eltype(u))]]
end


function d2dx(u,dx)
    """
    2nd order Central difference for 2nd degree derivative
    """
    return [[zero(eltype(u))]; (u[3:end] - 2.0.*u[2:end-1] + u[1:end-2]) ./ (dx^2); [zero(eltype(u))]]
end

## ODE description of the Physics:
function heat(u,p,t)
    # Model parameters
    a0, a1 = p
    dx,Nx = xtrs #[1.0,3.0,0.125,100]
    return 2.0*a0 .* u +  a1 .* d2dx(u, dx)
end

# Testing Solver on linear PDE
prob = ODEProblem(heat,u0,tspan,p)
sol = solve(prob,Tsit5(), dt=dt,saveat=t);

plot(x, sol.u[1], lw=3, label="t0", size=(800,500))
plot!(x, sol.u[end],lw=3, ls=:dash, label="tMax")

ps  = [0.1, 0.2];   # Initial guess for model parameters
function predict(θ)
    Array(solve(prob,Tsit5(),p=θ,dt=dt,saveat=t))
end

## Defining Loss function
function loss(θ)
    pred = predict(θ)
    l = predict(θ)  - sol
    return sum(abs2, l), pred # Mean squared error
end

l,pred   = loss(ps)
size(pred), size(sol), size(t) # Checking sizes

LOSS  = []                              # Loss accumulator
PRED  = []                              # prediction accumulator
PARS  = []                              # parameters accumulator

cb = function (θ,l,pred) #callback function to observe training
  display(l)
  append!(PRED, [pred])
  append!(LOSS, l)
  append!(PARS, [θ])
  false
end

cb(ps,loss(ps)...) # Testing callback function

# Let see prediction vs. Truth
scatter(sol[:,end], label="Truth", size=(800,500))
plot!(PRED[end][:,end], lw=2, label="Prediction")

res = DiffEqFlux.sciml_train(loss, ps, cb = cb)
@show res.u # returns [0.999999999613485, 0.9999999991343996]
```

## Step-by-step Description

### Load Packages

```julia
using DelimitedFiles,Plots
using DifferentialEquations, DiffEqFlux
```

### Parameters

First, we setup the 1-dimensional space over which our equations will be evaluated.
`x` spans **from 0.0 to 10.0** in steps of **0.01**; `t` spans **from 0.00 to 0.04** in
steps of **4.0e-5**.

```julia
# Problem setup parameters:
Lx = 10.0
x  = 0.0:0.01:Lx
dx = x[2] - x[1]
Nx = size(x)

u0 = exp.(-(x.-3.0).^2) # I.C

## Problem Parameters
p        = [1.0,1.0]    # True solution parameters
xtrs     = [dx,Nx]      # Extra parameters
dt       = 0.40*dx^2    # CFL condition
t0, tMax = 0.0 ,1000*dt
tspan    = (t0,tMax)
t        = t0:dt:tMax;
```

In plain terms, the quantities that were defined are:

- `x` (to `Lx`) spans the specified 1D space
- `dx` = distance between two points
- `Nx` = total size of space
- `u0` = initial condition
- `p` = true solution
- `xtrs` = convenient grouping of `dx` and `Nx` into Array
- `dt` = time distance between two points
- `t` (`t0` to `tMax`) spans the specified time frame
- `tspan` = span of `t`

### Auxiliary Functions

We then define two functions to compute the derivatives numerically. The **Central
Difference** is used in both the 1st and 2nd degree derivatives.

```julia
## Definition of Auxiliary functions
function ddx(u,dx)
    """
    2nd order Central difference for 1st degree derivative
    """
    return [[zero(eltype(u))] ; (u[3:end] - u[1:end-2]) ./ (2.0*dx) ; [zero(eltype(u))]]
end


function d2dx(u,dx)
    """
    2nd order Central difference for 2nd degree derivative
    """
    return [[zero(eltype(u))]; (u[3:end] - 2.0.*u[2:end-1] + u[1:end-2]) ./ (dx^2); [zero(eltype(u))]]
end
```

### Heat Differential Equation

Next, we setup our desired set of equations in order to define our problem.

```julia
## ODE description of the Physics:
function heat(u,p,t)
    # Model parameters
    a0, a1 = p
    dx,Nx = xtrs #[1.0,3.0,0.125,100]
    return 2.0*a0 .* u +  a1 .* d2dx(u, dx)
end
```

### Solve and Plot Ground Truth

We then solve and plot our partial differential equation. This is the true solution which we
will compare to further on.

```julia
# Testing Solver on linear PDE
prob = ODEProblem(heat,u0,tspan,p)
sol = solve(prob,Tsit5(), dt=dt,saveat=t);

plot(x, sol.u[1], lw=3, label="t0", size=(800,500))
plot!(x, sol.u[end],lw=3, ls=:dash, label="tMax")
```

### Building the Prediction Model

Now we start building our prediction model to try to obtain the values `p`. We make an
initial guess for the parameters and name it `ps` here. The `predict` function is a
non-linear transformation in one layer using `solve`. If unfamiliar with the concept,
refer to [here](https://julialang.org/blog/2019/01/fluxdiffeq/).

```julia
ps  = [0.1, 0.2];   # Initial guess for model parameters
function predict(θ)
    Array(solve(prob,Tsit5(),p=θ,dt=dt,saveat=t))
end
```

### Train Parameters

Training our model requires a **loss function**, an **optimizer** and a **callback
function** to display the progress.

#### Loss

We first make our predictions based on the current values of our parameters `ps`, then
take the difference between the predicted solution and the truth above. For the loss, we
use the **Mean squared error**.

```julia
## Defining Loss function
function loss(θ)
    pred = predict(θ)
    l = predict(θ)  - sol
    return sum(abs2, l), pred # Mean squared error
end

l,pred   = loss(ps)
size(pred), size(sol), size(t) # Checking sizes
```

#### Optimizer

The optimizers `ADAM` with a learning rate of 0.01 and `BFGS` are directly passed in
training (see below)

#### Callback

The callback function displays the loss during training. We also keep a history of the
loss, the previous predictions and the previous parameters with `LOSS`, `PRED` and `PARS`
accumulators.

```julia
LOSS  = []                              # Loss accumulator
PRED  = []                              # prediction accumulator
PARS  = []                              # parameters accumulator

cb = function (θ,l,pred) #callback function to observe training
  display(l)
  append!(PRED, [pred])
  append!(LOSS, l)
  append!(PARS, [θ])
  false
end

cb(ps,loss(ps)...) # Testing callback function
```

### Plotting Prediction vs Ground Truth

The scatter points plotted here are the ground truth obtained from the actual solution we
solved for above. The solid line represents our prediction. The goal is for both to overlap
almost perfectly when the PDE finishes its training and the loss is close to 0.

```julia
# Let see prediction vs. Truth
scatter(sol[:,end], label="Truth", size=(800,500))
plot!(PRED[end][:,end], lw=2, label="Prediction")
```

### Train

The parameters are trained using `sciml_train` and adjoint sensitivities.
The resulting best parameters are stored in `res` and `res.u` returns the
parameters that minimizes the cost function.

```julia
res = DiffEqFlux.sciml_train(loss, ps, cb = cb)
@show res.u # returns [0.999999999613485, 0.9999999991343996]
```

We successfully predict the final `ps` to be equal to **[0.999999999999975,
1.0000000000000213]** vs the true solution of `p` = **[1.0, 1.0]**

### Expected Output

```julia
153.74716386883014
153.74716386883014
150.31001476832154
146.91327105278128
143.55759898759374
140.24363496931753
136.97198347241257
133.7432151677673
130.55786524987215
127.4164319720337
124.31937540894337
121.26711645161134
118.26003603654628
115.29847461603427
112.3827318609633
109.51306659138356
106.68969692777314
103.9128006498965
101.18251574195561
98.4989411191655
95.8621374998964
93.27212842357801
90.7289013677808
88.23240896985287
85.7825703121191
83.37927225399383
81.02237079935475
78.71169247246975
76.44703568540336
74.22817209335733
72.05484791455291
69.92678520204167
67.84368308185877
65.80521891873633
63.81104944163126
61.860811797059554
59.95412455791812
58.090588663826914
56.26978832428055
54.491291863817686
52.75465253618253
51.05940929392087
49.405087540342564
47.79119984816457
46.217246667009626
44.68271701552145
43.18708916553295
41.729831330086824
40.310402328506555
38.928252289762675
37.58282331100446
36.27355015737786
34.99986094007708
33.76117780641769
32.55691762379305
31.386492661205562
30.249311268822595
29.144778544729924
28.07229699202965
27.031267166855155
26.0210883069299
25.041158938495613
24.09087747422764
23.169642780270983
22.276854715336583
21.411914664407295
20.57422602075309
19.76319467338999
18.978229434706996
18.218742481097735
17.48414972880479
16.773871221320032
16.087331469276343
15.423959781047255
14.78319057598673
14.164463661389682
13.567224508247984
12.990924508800399
12.435021204904853
11.898978515303417
11.382266943971572
10.884363779196345
10.404753276294088
9.942926832732251
9.49838314770057
9.070628379941386
8.659176278010788
8.263548334737965
7.883273889583058
7.517890250788576
7.1669427976429585
6.829985075319055
6.506578881124348
6.19629433688754
5.898709957062298
5.613412692266443
5.339997993203038
5.078069839645422
4.827240754206443
4.587131834698446
4.357372763056912
4.357372763056912
4.137601774726927
1.5254536025963588
0.0023707487489687726
4.933077457357198e-7
8.157805551380282e-14
1.6648677430325974e-16
res.u = [0.999999999999975, 1.0000000000000213]
2-element Array{Float64,1}:
 0.999999999999975
 1.0000000000000213
```
