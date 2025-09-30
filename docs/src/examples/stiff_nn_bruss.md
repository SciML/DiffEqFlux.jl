# Learning the Brusselator Equation with a Universal Differential Equation (UDE)

This document walks through an example of using a Universal Differential Equation (UDE) to learn the dynamics of a reaction-diffusion system known as the Brusselator.

### Context: The Brusselator and UDEs

The **Brusselator** is a partial differential equation (PDE) that models a theoretical autocatalytic chemical reaction. It describes how the concentrations of two chemical species evolve over space and time, governed by two main processes:
1.  **Reaction**: The species interact with each other, changing their concentrations locally.
2.  **Diffusion**: The species spread out over the spatial domain.

A **Universal Differential Equation (UDE)** is a hybrid modeling approach that merges known physical laws with machine learning. The core idea is to encode the parts of the system you understand (e.g., diffusion) directly into the equations and use a neural network to learn the parts you don't (e.g., the complex reaction kinetics).

In this example, we will:
1.  Generate "ground truth" data by solving the full, known Brusselator PDE.
2.  Define a UDE where the diffusion term is explicitly coded, but the reaction term is replaced by a neural network.
3.  Train the neural network's parameters by requiring the UDE's solution to match the ground truth data.
4.  Visualize the results to confirm that our UDE has successfully learned the unknown reaction dynamics.

This showcases the power of scientific machine learning (SciML) to discover governing equations from data.

---

### 1. Problem Setup and Dependencies

First, we import the necessary Julia libraries. We'll use `DifferentialEquations.jl` for solving ODEs, `Lux.jl` for the neural network, `Optimization.jl` for training, and `Plots.jl` for visualization. We then define the simulation constants, such as grid size and simulation time.


```@example stiff_bruss
using OrdinaryDiffEq, DifferentialEquations
using LinearSolve
using SciMLSensitivity
using Lux
using Optimization, OptimizationOptimisers
using StaticArrays
using NNlib
using Random, Zygote
using Plots, Statistics
using Base.Threads
using ComponentArrays
using ReverseDiff

# We disable the default plot saving and display them directly.
default(show = true)

# 1. Problem Setup: Constants, Grid, and Initial Conditions

# -- Simulation Parameters --
const N         = 32
const TEND      = 11.5f0
const MAXITERS  = 200
const RTOL_REF  = 1f-6
const ATOL_REF  = 1f-6

# -- Grid and Discretization --
const xyd = range(0.0f0, 1.0f0, length = N) # Spatial domain
const dx = step(xyd)                        # Spatial step size

"""
    limit(a::Int, N::Int)

Enforces periodic boundary conditions by wrapping indices around the grid.
If an index `a` goes past the boundary (1 or N), it wraps to the other side.
"""
@inline limit(a::Int, N::Int) = a == N + 1 ? 1 : (a == 0 ? N : a)

"""
    brusselator_f(x, y, t)

A forcing term for the Brusselator equation, which is active in a circular
region for t ≥ 1.1.
"""
@inline brusselator_f(x, y, t) = (((x - 0.3)^2 + (y - 0.6)^2) <= 0.1^2) * (t >= 1.1) * 5.0

"""
    init_u0(xyd)

Generates the initial condition `u0` for the two species on the grid.
"""
function init_u0(xyd)
    N = length(xyd)
    u = zeros(Float32, N, N, 2)
    @inbounds for I in CartesianIndices((N, N))
        x = Float32(xyd[I[1]]); y = Float32(xyd[I[2]])
        u[I, 1] = 22.0f0 * (y * (1.0f0 - y))^(3.0f0 / 2.0f0)
        u[I, 2] = 27.0f0 * (x * (1.0f0 - x))^(3.0f0 / 2.0f0)
    end
    return u
end

# Initialize the state vector `u0`
u0 = init_u0(xyd)
```

### 2. Generating the Reference Solution (Ground Truth)
To train our UDE, we need data to learn from. We generate this by solving the full Brusselator PDE with its known equations. This solution will serve as our "ground truth" that we will try to replicate with the UDE. The rhs_ref! function defines the complete dynamics, including both diffusion and reaction terms.

```@example stiff_bruss
# 2. Reference Solution (Ground Truth)

# Here, we solve the full PDE with the known reaction terms to generate
# the data we will use to train our neural network.

# -- Brusselator PDE Parameters --
const α = 10.0f0
const αdx2 = α / dx^2

"""
    rhs_ref!(du, u, p, t)

The right-hand side (RHS) function for the Brusselator PDE, including both the
diffusion (Laplacian) and the known reaction terms. This defines the true dynamics.
"""
println("Stage 1/4: Generating reference solution...")
function rhs_ref!(du, u, p, t)
    A, B, alpha_val = p
    alpha_val = alpha_val / dx^2
    @inbounds for I in CartesianIndices((N, N))
        i, j = Tuple(I)
        x, y = xyd[I[1]], xyd[I[2]]
        ip1, im1, jp1, jm1 = limit(i + 1, N), limit(i - 1, N), limit(j + 1, N), limit(j - 1, N)
        du[i, j, 1] = alpha_val * (u[im1, j, 1] + u[ip1, j, 1] + u[i, jp1, 1] + u[i, jm1, 1] - 4u[i, j, 1]) +
                               B + u[i, j, 1]^2 * u[i, j, 2] - (A + 1) * u[i, j, 1] +
                               brusselator_f(x, y, t)
        du[i, j, 2] = alpha_val * (u[im1, j, 2] + u[ip1, j, 2] + u[i, jp1, 2] + u[i, jm1, 2] - 4u[i, j, 2]) +
                               A * u[i, j, 1] - u[i, j, 1]^2 * u[i, j, 2]
    end
end

p_ref = (3.4, 1.0, 10.0)
prob_ref = ODEProblem(rhs_ref!, u0, (0.0, TEND), p_ref)
sol_ref  = solve(prob_ref, KenCarp47(linsolve=KrylovJL_GMRES());
                 saveat=0.0:0.5:TEND, reltol=RTOL_REF, abstol=ATOL_REF, progress=true)

const Yref = Array(sol_ref)
const ts = sol_ref.t
const mean_true = [mean(Yref[:,:,1,i]) for i in 1:size(Yref, 4)]
println("Ground truth generated. Size: ", size(Yref))
```

### 3. Defining the Neural Network
Next, we define the neural network architecture that will learn the unknown reaction term. The model is a simple multi-layer perceptron with a custom SigmaLayer that applies a learnable, exponentially decaying weight to its inputs. We also create helper functions (flatten_ps, to_ps) to convert the network's parameters between Lux's structured format and the flat vector format required by the optimizer.

```@example stiff_bruss
# 3. Neural Network (UDE Component)

# This section defines the neural network architecture that will learn the
# unknown reaction term.

import LuxCore: initialparameters, initialstates
using Random, Lux, ComponentArrays

const H = 16

# --- 1. Define the Custom Neural Network Layer ---
"""
    SigmaLayerNN{M} <: Lux.AbstractLuxLayer

A custom Lux layer that contains an internal neural network (`net`). This
internal net learns the stiffening values (`σ`) which are then applied to the layer's input.
"""
struct SigmaLayerNN{M} <: Lux.AbstractLuxLayer
    net::M
end

"""
    SigmaLayerNN(H::Int)

Constructor for the `SigmaLayerNN`. It initializes the internal neural network.
"""
function SigmaLayerNN(H::Int)
    net = Dense(H => H, tanh)
    return SigmaLayerNN(net)
end

# --- 2. Define How Lux Interacts with the Custom Layer ---

# Explicitly tell Lux how to get the parameters for the inner network.
function initialparameters(rng::AbstractRNG, ℓ::SigmaLayerNN)
    return (net = initialparameters(rng, ℓ.net),)
end

# Explicitly tell Lux how to get the state for the inner network.
function initialstates(rng::AbstractRNG, ℓ::SigmaLayerNN)
    return (net = initialstates(rng, ℓ.net),)
end

# --- 3. Define the Layer's Forward Pass ---

"""
    (ℓ::SigmaLayerNN)(z, ps, st)

The forward pass for the `SigmaLayerNN`. It takes an input `z`, passes it
through the internal net to get the stiffening values `σ`, and then applies
those values to `z`.
"""
function (ℓ::SigmaLayerNN)(z, ps, st)
    # Get the raw output from the internal network
    σ_raw, st_net = ℓ.net(z, ps.net, st.net)

    # Apply the sigmoid function to ensure stiffening values are positive
    σ = 1.0f0 ./ (1.0f0 .+ exp.(-σ_raw))

    # Apply the learned stiffening values, handling batch dimensions if present
    if ndims(z) == 1
        z = z .* σ
    else
        z = z .* reshape(σ, :, 1)
    end

    # Return the result and the updated state of the internal network
    return z, (net = st_net,)
end

# --- 4. Build and Initialize the Full Model ---

# Create the full model by chaining the layers together
model = Chain(
    Dense(2 => H, tanh),
    SigmaLayerNN(H),
    Dense(H => 2)
)

# Initialize the model's parameters (ps0) and state (st0)
rng = Random.default_rng()
ps0, st0 = Lux.setup(rng, model)

# Define the constant state for training
const ST = st0

# Create the initial flat parameter vector using ComponentArrays
θ0 = ComponentArray(ps0)
```

### 4. Constructing the Universal Differential Equation (UDE)
Here is the core of the UDE. The rhs_ude! function defines the hybrid dynamics. It explicitly calculates the diffusion term (the known physics) and calls the neural network to approximate the reaction term (the unknown physics). This function is then used to create an ODEProblem that can be solved and differentiated.

```@example stiff_bruss
# 4. Universal Differential Equation (UDE)

# The UDE combines the known physics (diffusion) with the neural network.

const COUT = 5.0f0 # Clamp NN output to prevent explosions during training

"""
    rhs_ude!(du, u, θ_vec, t)

The right-hand side (RHS) function for the Universal Differential Equation (UDE).

This function combines known physical laws (diffusion) with a neural network that learns
the unknown reaction dynamics. It operates over a 2D grid in a single loop.
"""
function rhs_ude!(du, u, θ_vec, t)
    Tz = eltype(u)
    loop_body = I -> begin
        i,j = Tuple(I)
        x = Float32(xyd[i]); y = Float32(xyd[j])
        u1 = u[i,j,1]; v1 = u[i,j,2]
        lap_u = u[limit(i-1,N),j,1]+u[limit(i+1,N),j,1]+u[i,limit(j+1,N),1]+u[i,limit(j-1,N),1]-4f0*u1
        lap_v = u[limit(i-1,N),j,2]+u[limit(i+1,N),j,2]+u[i,limit(j+1,N),2]+u[i,limit(j-1,N),2]-4f0*v1
        x_in = Tz[u1, v1]
        ŷ, _ = model(x_in, θ_vec, ST)
        y1 = clamp(ŷ[1], -COUT, COUT)
        y2 = clamp(ŷ[2], -COUT, COUT)
        du[i,j,1] = αdx2*lap_u + y1 + brusselator_f(x,y,t)
        du[i,j,2] = αdx2*lap_v + y2
    end
    @inbounds @threads for I in CartesianIndices((N,N))
        loop_body(I)
    end
    nothing
end

# Define the ODE problem for the UDE, passing the NN parameters `θ0`
prob_ude = ODEProblem(rhs_ude!, u0, (0.0, TEND), θ0)
```

### 5. Training the UDE
With the UDE defined, we can now train it. The loss function solves the UDE with the current neural network parameters and computes the mean squared error against the reference data. We use Optimization.jl with the Adam optimizer to minimize this loss. SciMLSensitivity.jl provides the magic to efficiently compute gradients of the loss function with respect to the network parameters, even though the parameters are inside a differential equation solver.

```@example stiff_bruss
# 5. Training the UDE

println("\nStage 2/4: Setting up loss function and optimizer...")

# Define the sensitivity algorithm for calculating gradients efficiently
sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP())

"""
    loss(θ_vec)

Computes the mean squared error between the UDE solution (using parameters `θ_vec`)
and the ground truth solution `Yref`.
"""
function loss(θ_vec)
    sol = solve(remake(prob_ude; p=θ_vec), KenCarp47(linsolve=LinearSolve.KrylovJL_GMRES());
                saveat=ts, reltol=1f-4, abstol=1f-4, save_everystep=false, sensealg=sensealg)
    Y = Array(sol)
    if size(Y) != size(Yref)
        return Inf32
    end
    sum(abs2, Y .- Yref) / length(Yref)
end

# -- Setup the optimization problem --
optf    = OptimizationFunction((θ, _)->loss(θ), AutoReverseDiff())
optprob = OptimizationProblem(optf, θ0)

# -- Define a callback to monitor training progress --
println("Stage 3/4: Starting training...")
k_iter = 0
function cb(θ, f_val)
    global k_iter += 1
    if k_iter % 5 == 0
        println("  Iter: $(k_iter) \t Loss: $(round(f_val, digits=6))")
        flush(stdout)
    end
    # Return false to continue optimization
    return false
end

# -- Run the optimization --
solopt   = solve(optprob, Optimisers.Adam(1e-2); maxiters=MAXITERS, callback=cb)
θ★ = solopt.u # The optimal parameters

println("Training finished.")
```

### 6. Evaluation and Visualization
Finally, after training, we evaluate the performance of our UDE. We solve the UDE one last time using the final, optimized parameters (θ★). We then create two plots to compare the UDE's solution to the ground truth:

A heatmap showing the spatial concentration of one species at the final time point.

A time-series plot showing the evolution of the mean concentration over the entire simulation.

If the training was successful, the UDE's output should closely match the true simulation.

```@example stiff_bruss
# 6. Evaluation and Visualization

println("\nStage 4/4: Evaluating final model and generating plots...")

# Solve the UDE one last time with the optimized parameters `θ★`
sol_ude = solve(remake(prob_ude; p = θ★), KenCarp47(linsolve = KrylovJL_GMRES());
    saveat = ts, reltol = 1e-6, abstol = 1e-6, save_everystep = false)

# Calculate the final relative mean squared error
final_loss = sum(abs2, Array(sol_ude) .- Yref) / sum(abs2, Yref)
println("Done. Final relative MSE = ", final_loss)

# -- Create comparison plots --

# 1. Heatmap comparison of the final state
final_state_true = Yref[:,:,1,end]
final_state_ude = sol_ude_final.u[end][:, :, 1]

p1 = heatmap(final_state_true, title="True Simulation (t=$(TEND))")
p2 = heatmap(final_state_ude, title="Final SNN-UDE (it=$(k_iter))")
comparison_plot = plot(p1, p2, layout=(1, 2), size=(900, 400))
display(comparison_plot)

# 2. Time series comparison of the mean concentration
mean_ude = [mean(u[:,:,1]) for u in sol_ude_final.u]
metric_plot = plot(ts, mean_true, label="True Simulation", lw=2, xlabel="Time (t)", ylabel="Mean Concentration", title="Model Performance (Final)")
plot!(ts, mean_ude, label="SNN-UDE Prediction", lw=2, linestyle=:dash)
display(metric_plot)

println("\nPlots are displayed.")
```
