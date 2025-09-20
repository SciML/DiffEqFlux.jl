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
using Random, Zygote
using Plots, Statistics

# We disable the default plot saving and display them directly.
default(show = true)

# 1. Problem Setup: Constants, Grid, and Initial Conditions

# -- Simulation Parameters --
const N = 16            # Grid size will be N x N
const TEND = 3.0f0      # End time of the simulation
const MAXITERS = 50     # Number of training iterations for the optimizer
const H = 16            # Hidden layer size for the neural network

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
@inline function brusselator_f(x, y, t)
    forcing = (((x - 0.3f0)^2 + (y - 0.6f0)^2) <= 0.1f0^2) && (t >= 1.1f0)
    return forcing ? 5.0f0 : 0.0f0
end

"""
    init_u0(xyd)

Generates the initial condition `u0` for the two species on the grid.
"""
function init_u0(xyd)
    N = length(xyd)
    u = zeros(Float32, N, N, 2)
    @inbounds for I in CartesianIndices((N, N))
        x = xyd[I[1]]
        y = xyd[I[2]]
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
const A = 3.4f0
const B = 1.0f0
const α = 10.0f0
const αdx2 = α / dx^2

"""
    rhs_ref!(du, u, p, t)

The right-hand side (RHS) function for the Brusselator PDE, including both the
diffusion (Laplacian) and the known reaction terms. This defines the true dynamics.
"""
function rhs_ref!(du, u, p, t)
    @inbounds for I in CartesianIndices((N, N))
        i, j = Tuple(I)
        x, y = xyd[i], xyd[j]

        # Neighbor indices with periodic boundaries
        ip1 = limit(i + 1, N)
        im1 = limit(i - 1, N)
        jp1 = limit(j + 1, N)
        jm1 = limit(j - 1, N)

        u1 = u[i, j, 1]
        v1 = u[i, j, 2]

        # Discretized Laplacian for diffusion
        lap_u = (u[im1, j, 1] + u[ip1, j, 1] + u[i, jm1, 1] + u[i, jp1, 1] - 4.0f0 * u1)
        lap_v = (u[im1, j, 2] + u[ip1, j, 2] + u[i, jm1, 2] + u[i, jp1, 2] - 4.0f0 * v1)

        # Known Brusselator reaction terms
        reaction1 = B + u1 * u1 * v1 - (A + 1.0f0) * u1
        reaction2 = A * u1 - u1 * u1 * v1

        # Combine diffusion, reaction, and forcing term
        du[i, j, 1] = αdx2 * lap_u + reaction1 + brusselator_f(x, y, t)
        du[i, j, 2] = αdx2 * lap_v + reaction2
    end
    return nothing
end

println("Stage 1/4: Generating reference solution...")
prob_ref = ODEProblem(rhs_ref!, u0, (0.0f0, TEND))
sol_ref = solve(prob_ref, KenCarp47(linsolve = KrylovJL_GMRES());
    saveat = 0.0f0:0.5f0:TEND, reltol = 1e-5, abstol = 1e-5,
    save_everystep = false, progress = true)

# Store the reference solution and time points for training and comparison
Yref = Array(sol_ref)
ts = sol_ref.t
println("Reference solution generated.")
```

### 3. Defining the Neural Network
Next, we define the neural network architecture that will learn the unknown reaction term. The model is a simple multi-layer perceptron with a custom SigmaLayer that applies a learnable, exponentially decaying weight to its inputs. We also create helper functions (flatten_ps, to_ps) to convert the network's parameters between Lux's structured format and the flat vector format required by the optimizer.

```@example stiff_bruss
# 3. Neural Network (UDE Component)

# This section defines the neural network architecture that will learn the
# unknown reaction term.

"""
    SigmaLayer

A custom Lux layer that applies a learnable, exponentially decaying weight
to the activations. The decay rate `p` is a learnable parameter.
"""
struct SigmaLayer <: Lux.AbstractLuxLayer end

# Initialize the layer's parameters and state
Lux.initialparameters(rng::AbstractRNG, ::SigmaLayer) = (p = 2.0f0,)
Lux.initialstates(rng::AbstractRNG, ::SigmaLayer) = NamedTuple()

# Define the layer's forward pass
function (ℓ::SigmaLayer)(z, ps, st)
    H = size(z, 1) # Height (number of neurons)
    Tz = eltype(z)
    # Use softplus to ensure the decay rate `σ` is positive
    σ = NNlib.softplus(Tz(ps.p))
    decay = exp.(-σ .* (1:H))

    # Apply decay, broadcasting over a batch if necessary
    z = z .* reshape(decay, H, ntuple(Returns(1), ndims(z) - 1)...)
    return z, st
end

# -- Define the full neural network model --
# The model takes the concentrations of the two species `[u, v]` as input
# and outputs the predicted reaction terms `[reaction1, reaction2]`.
model = Chain(Dense(2 => H, tanh), SigmaLayer(), Dense(H => 2))

# Initialize model parameters (ps) and state (st)
rng = Random.default_rng()
ps0, st0 = Lux.setup(rng, model)
const ST = st0 # State is constant during training

"""
    flatten_ps(ps)

Converts the nested Lux parameter structure `ps` into a flat `Vector{Float32}`.
This is required by the `Optimization.jl` interface.
"""
function flatten_ps(ps)::Vector{Float32}
    w1 = vec(ps.layer_1.weight)
    b1 = ps.layer_1.bias
    p = ps.layer_2.p
    w3 = vec(ps.layer_3.weight)
    b3 = ps.layer_3.bias
    return vcat(w1, b1, p, w3, b3)
end

"""
    to_ps(θ::AbstractVector)

Reconstructs the Lux parameter structure `ps` from a flat vector `θ`.
"""
function to_ps(θ::AbstractVector)
    # Calculate expected length to catch errors if model architecture changes
    expected_len = (2 * H) + H + 1 + (H * 2) + 2 # W1 + b1 + p2 + W3 + b3
    @assert length(θ) == expected_len "Incorrect parameter vector length."

    T = eltype(θ)
    i = 1
    # Layer 1: Dense
    w1_end = i + 2 * H - 1
    w1 = reshape(θ[i:w1_end], H, 2)
    i = w1_end + 1
    # Layer 1: Bias
    b1_end = i + H - 1
    b1 = θ[i:b1_end]
    i = b1_end + 1
    # Layer 2: SigmaLayer
    p2 = θ[i]
    i += 1
    # Layer 3: Dense
    w3_end = i + 2 * H - 1
    w3 = reshape(θ[i:w3_end], 2, H)
    i = w3_end + 1
    # Layer 3: Bias
    b3 = θ[i:end]

    return (layer_1 = (weight = T.(w1), bias = T.(b1)),
        layer_2 = (p = T(p2),),
        layer_3 = (weight = T.(w3), bias = T.(b3)))
end

# Get the initial flat parameter vector
θ0 = flatten_ps(ps0)
```

### 4. Constructing the Universal Differential Equation (UDE)
Here is the core of the UDE. The rhs_ude! function defines the hybrid dynamics. It explicitly calculates the diffusion term (the known physics) and calls the neural network to approximate the reaction term (the unknown physics). This function is then used to create an ODEProblem that can be solved and differentiated.

```@example stiff_bruss
# 4. Universal Differential Equation (UDE)

# The UDE combines the known physics (diffusion) with the neural network.

const COUT = 5.0f0 # Clamp NN output to prevent explosions during training

"""
    rhs_ude!(du, u, θ_vec, t)

The RHS function for the UDE. It computes the diffusion term analytically and
uses the neural network `model` (with parameters `θ_vec`) to approximate the
reaction term.
"""
function rhs_ude!(du, u, θ_vec, t)
    psθ = to_ps(θ_vec) # Reconstruct NN parameters for Lux
    Tz = eltype(u)

    @inbounds for I in CartesianIndices((N, N))
        i, j = Tuple(I)
        x, y = xyd[i], xyd[j]

        # Neighbor indices with periodic boundaries
        ip1 = limit(i + 1, N)
        im1 = limit(i - 1, N)
        jp1 = limit(j + 1, N)
        jm1 = limit(j - 1, N)

        u1 = u[i, j, 1]
        v1 = u[i, j, 2]

        # Part 1: Known Physics (Diffusion)
        lap_u = (u[im1, j, 1] + u[ip1, j, 1] + u[i, jm1, 1] + u[i, jp1, 1] - 4.0f0 * u1)
        lap_v = (u[im1, j, 2] + u[ip1, j, 2] + u[i, jm1, 2] + u[i, jp1, 2] - 4.0f0 * v1)

        # Part 2: Unknown Physics (Learned by NN)
        # Input to the NN is the state [u1, v1] at a single grid point
        nn_input = Tz[u1, v1]
        reaction_pred, _ = model(nn_input, psθ, ST)

        # Clamp the NN output for stability
        y1 = clamp(reaction_pred[1], -COUT, COUT)
        y2 = clamp(reaction_pred[2], -COUT, COUT)

        # Combine known physics, learned reaction, and forcing term
        du[i, j, 1] = αdx2 * lap_u + y1 + brusselator_f(x, y, t)
        du[i, j, 2] = αdx2 * lap_v + y2
    end
    return nothing
end

# Define the ODE problem for the UDE, passing the NN parameters `θ0`
prob_ude = ODEProblem(rhs_ude!, u0, (0.0f0, TEND), θ0)
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
    # Solve the UDE with the current parameter vector
    sol = solve(remake(prob_ude; p = θ_vec), KenCarp47(linsolve = KrylovJL_GMRES());
        saveat = ts, reltol = 1e-4, abstol = 1e-4,
        save_everystep = false, sensealg = sensealg)

    # Return Inf if the solver failed to produce a solution of the correct size
    if size(sol) != size(Yref)
        return Inf32
    end

    # Return the mean squared error
    Y = Array(sol)
    return sum(abs2, Y .- Yref) / length(Yref)
end

# -- Setup the optimization problem --
optf = OptimizationFunction((θ, _) -> loss(θ), AutoZygote())
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
solopt = solve(optprob, Optimisers.Adam(1e-2); maxiters = MAXITERS, callback = cb)
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
    saveat = ts, reltol = 1e-5, abstol = 1e-5, save_everystep = false)

# Calculate the final relative mean squared error
final_loss = sum(abs2, Array(sol_ude) .- Yref) / sum(abs2, Yref)
println("Done. Final relative MSE = ", final_loss)

# -- Create comparison plots --

# 1. Heatmap comparison of the final state
final_state_true = sol_ref.u[end][:, :, 1]
final_state_ude = sol_ude.u[end][:, :, 1]

p1 = heatmap(final_state_true, title = "True Simulation (t=$(TEND))")
p2 = heatmap(final_state_ude, title = "Final UDE (it=$(k_iter))")
comparison_plot = plot(p1, p2, layout = (1, 2), size = (900, 400))
display(comparison_plot)

# 2. Time series comparison of the mean concentration
mean_true = [mean(u[:, :, 1]) for u in sol_ref.u]
mean_ude = [mean(u[:, :, 1]) for u in sol_ude.u]

metric_plot = plot(ts, mean_true, label = "True Simulation", lw = 2,
    xlabel = "Time (t)", ylabel = "Mean Concentration",
    title = "Model Performance vs. Ground Truth")
plot!(ts, mean_ude, label = "UDE Prediction", lw = 2, linestyle = :dash)
display(metric_plot)

println("\nPlots are displayed.")
```
