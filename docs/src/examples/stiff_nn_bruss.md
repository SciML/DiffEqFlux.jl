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
using Optimisers
using NNlib
using Random, Zygote
using Plots, Statistics
using Base.Threads
using ComponentArrays
using JLD2
using Dates
# We disable the default plot saving and display them directly.

# 1. Problem Setup: Constants, Grid, and Initial Conditions

# -- Simulation Parameters --
const N         = 16
const TEND      = 12.0f0
const MAXITERS  = 200
const RTOL_REF  = 1f-6
const ATOL_REF  = 1f-6

# -- Grid and Discretization --
const xyd = range(0f0, stop=1f0, length=N)
dx = step(xyd)

@inline limit(a::Int, N::Int) = a == N + 1 ? 1 : (a == 0 ? N : a)
@inline brusselator_f(x, y, t) =
    (((x - 0.3f0)^2 + (y - 0.6f0)^2) <= 0.1f0^2) * (t >= 1.1f0) * 5.0f0

function init_u0(xyd)
    N = length(xyd)
    u = zeros(Float32, N, N, 2)
    @inbounds for I in CartesianIndices((N,N))
        x = Float32(xyd[I[1]]); y = Float32(xyd[I[2]])
        u[I,1] = 22f0 * (y*(1f0-y))^(3f0/2f0)
        u[I,2] = 27f0 * (x*(1f0-x))^(3f0/2f0)
    end
    return u
end
const u0   = init_u0(xyd)
const α    = 10.0f0
const αdx2 = α / (dx*dx)
```

### 2. Generating the Reference Solution (Ground Truth)
To train our UDE, we need data to learn from. We generate this by solving the full Brusselator PDE with its known equations. This solution will serve as our "ground truth" that we will try to replicate with the UDE. The rhs_ref! function defines the complete dynamics, including both diffusion and reaction terms.

```@example stiff_bruss
println("Stage 1/4: Ground truth not found. Generating..."); flush(stdout)

function rhs_ref!(du, u, p, t)
    A, B, αval = p
    αval_dx2 = αval / (dx*dx)
    @inbounds for I in CartesianIndices((N, N))
        i, j = Tuple(I)
        x, y = xyd[i], xyd[j]
        ip1, im1 = limit(i+1,N), limit(i-1,N)
        jp1, jm1 = limit(j+1,N), limit(j-1,N)
        u1 = u[i,j,1]; v1 = u[i,j,2]
        lap_u = u[im1,j,1] + u[ip1,j,1] + u[i,jp1,1] + u[i,jm1,1] - 4f0*u1
        lap_v = u[im1,j,2] + u[ip1,j,2] + u[i,jp1,2] + u[i,jm1,2] - 4f0*v1
        du[i,j,1] = αval_dx2*lap_u + B + u1^2*v1 - (A+1f0)*u1 + brusselator_f(x,y,t)
        du[i,j,2] = αval_dx2*lap_v + A*u1 - u1^2*v1
    end
    return nothing
end

p_ref    = (3.4f0, 1.0f0, 10.0f0)
prob_ref = ODEProblem(rhs_ref!, u0, (0.0f0, TEND), p_ref)
sol_ref  = solve(prob_ref, KenCarp47(linsolve=LinearSolve.KrylovJL_GMRES());
                    saveat=0.0f0:0.5f0:TEND, reltol=RTOL_REF, abstol=ATOL_REF, progress=true)

global Yref = Array(sol_ref)
global ts   = sol_ref.t

const mean_true = [mean(Yref[:,:,1,i]) for i in 1:size(Yref,4)]
println("Ground truth loaded. Size: ", size(Yref))
```

### 3. Defining the Neural Network
Next, we define the neural network architecture that will learn the unknown reaction term. The model is a simple multi-layer perceptron with a custom SigmaLayer that applies a learnable, exponentially decaying weight to its inputs. We also create helper functions (flatten_ps, to_ps) to convert the network's parameters between Lux's structured format and the flat vector format required by the optimizer.

```@example stiff_bruss
# 3. Neural Network (UDE Component)

# This section defines the neural network architecture that will learn the
# unknown reaction term.

import LuxCore: initialparameters, initialstates

"""
SigmaDiag: diagonal Σ in state space (m = length(x)), PSD with mode-indexed decay.
"""
struct SigmaDiag <: Lux.AbstractLuxLayer
    kind::Symbol  # :logistic or :exp
end
function initialparameters(::AbstractRNG, Σ::SigmaDiag)
    if Σ.kind === :exp
        return (β = 1.0f0,)
    else
        return (a = -3.0f0, b = 4.0f0)  # logistic: σ_k = σ(a*k + b)
    end
end
initialstates(::AbstractRNG, ::SigmaDiag) = NamedTuple()
function (Σ::SigmaDiag)(x::AbstractVector{T}, ps, st) where {T}
    k = T.(1:length(x))
    σ = Σ.kind === :exp      ? exp.(-ps.β .* k) :
        Σ.kind === :logistic ? one(T) ./(one(T) .+ exp.(-(ps.a .* k .+ ps.b))) :
        error("Unknown Σ.kind=$(Σ.kind)")
    return σ .* x, st
end

"""
StiffUVSigma: F(x) = U( Σ( V(x) ) )   with U,V: ℝ^m→ℝ^m and diagonal Σ.
"""
struct StiffUVSigma{V,U,S} <: Lux.AbstractLuxLayer
    V::V
    Σ::S
    U::U
end
function initialparameters(rng::AbstractRNG, M::StiffUVSigma)
    return (V = initialparameters(rng, M.V),
            Σ = initialparameters(rng, M.Σ),
            U = initialparameters(rng, M.U))
end
function initialstates(rng::AbstractRNG, M::StiffUVSigma)
    return (V = initialstates(rng, M.V),
            Σ = initialstates(rng, M.Σ),
            U = initialstates(rng, M.U))
end
function (M::StiffUVSigma)(x, ps, st)
    yV, stV = M.V(x, ps.V, st.V)
    yΣ, stΣ = M.Σ(yV, ps.Σ, st.Σ)
    yU, stU = M.U(yΣ, ps.U, st.U)
    return yU, (V = stV, Σ = stΣ, U = stU)
end

# Model: U( Σ V(x) ) with m=2 state; H is width in U/V
const H = 16
const STIFF_KIND = Symbol(get(ENV, "STIFF_KIND", "logistic"))  # :logistic | :exp

Random.seed!(1234)
Vnet = Chain(Dense(2 => H, tanh), Dense(H => 2))
Σlay = SigmaDiag(STIFF_KIND)
Unet = Chain(Dense(2 => H, tanh), Dense(H => 2))
model = StiffUVSigma(Vnet, Σlay, Unet)

rng   = Random.default_rng()
ps0, st0 = Lux.setup(rng, model)
const ST = st0
θ0 = ComponentArray(ps0)
```

### 4. Constructing the Universal Differential Equation (UDE)
Here is the core of the UDE. The rhs_ude! function defines the hybrid dynamics. It explicitly calculates the diffusion term (the known physics) and calls the neural network to approximate the reaction term (the unknown physics). This function is then used to create an ODEProblem that can be solved and differentiated.

```@example stiff_bruss
const RESIDUAL   = true
const COUT       = 8.0
const RESID_COUT = 0.7 
const A0         = 3.4
const B0         = 1.0

function rhs_ude!(du, u, θ_vec, t)
    Tz = eltype(u)
    @inbounds for I in CartesianIndices((N,N))
        i, j = Tuple(I); x = xyd[i]; y = xyd[j]
        u1 = u[i,j,1]; v1 = u[i,j,2]
        lap_u = u[limit(i-1,N),j,1] + u[limit(i+1,N),j,1] +
                u[i,limit(j+1,N),1] + u[i,limit(j-1,N),1] - 4f0*u1
        lap_v = u[limit(i-1,N),j,2] + u[limit(i+1,N),j,2] +
                u[i,limit(j+1,N),2] + u[i,limit(j-1,N),2] - 4f0*v1
        ŷ, _ = model(Tz[u1, v1], θ_vec, ST)
        if RESIDUAL
            r1 = B0 + u1^2*v1 - (A0 + 1f0)*u1
            r2 = A0*u1 - u1^2*v1
            δ1 = clamp(ŷ[1], -RESID_COUT, RESID_COUT)
            δ2 = clamp(ŷ[2], -RESID_COUT, RESID_COUT)
            du[i,j,1] = αdx2*lap_u + (r1 + δ1) + brusselator_f(x,y,t)
            du[i,j,2] = αdx2*lap_v + (r2 + δ2)
        else
            y1 = clamp(ŷ[1], -COUT, COUT); y2 = clamp(ŷ[2], -COUT, COUT)
            du[i,j,1] = αdx2*lap_u + y1 + brusselator_f(x,y,t)
            du[i,j,2] = αdx2*lap_v + y2
        end
    end
    return nothing
end

prob_ude = ODEProblem(rhs_ude!, u0, (0.0f0, TEND), θ0)
```

### 5. Training the UDE
With the UDE defined, we can now train it. The loss function solves the UDE with the current neural network parameters and computes the mean squared error against the reference data. We use Optimization.jl with the Adam optimizer to minimize this loss. SciMLSensitivity.jl provides the functionality to efficiently compute gradients of the loss function with respect to the network parameters, even though the parameters are inside a differential equation solver.

```@example stiff_bruss
println("Stage 2/4: Set loss + optimizer …"); flush(stdout)
sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP())

const ALG_FINAL   = TRBDF2(autodiff=false)
const FINAL_RTOL  = 1f-4
const FINAL_ATOL  = 1f-4
const FINAL_DTMAX = 0.1f0

# ---- Zygote-safe constants used in loss ----
const LAMBDA = 3e-3

# Curriculum in time for training only
const TRAIN_FRAC = 1.0
const TRAIN_IDX  = 1:clamp(ceil(Int, TRAIN_FRAC * length(ts)), 1, length(ts))
const ts_train   = ts[TRAIN_IDX]
const Yref_train = Yref[:,:,:,TRAIN_IDX]
const W_WEIGHTS  = reshape(1 .+ collect(ts_train)./maximum(ts_train), 1,1,1,length(TRAIN_IDX))

function loss(θ_vec)
    sol = solve(remake(prob_ude; p=θ_vec), TRBDF2(autodiff=false);
                saveat=ts_train, reltol=1f-4, abstol=1f-4,
                save_everystep=false, sensealg=sensealg)
    Y = Array(sol)
    if size(Y) != size(Yref_train)
        return Inf32
    end
    data_mse = sum(abs2, (Y .- Yref_train) .* W_WEIGHTS) / length(Yref_train)
    reg = LAMBDA * sum(abs2, θ_vec)
    return data_mse + reg
end

optf    = OptimizationFunction((θ, _)->loss(θ), AutoZygote())
optprob = OptimizationProblem(optf, θ0)

println("Stage 3/4: Training …"); flush(stdout)
k_iter = 0
function cb(state, L)
    global k_iter += 1
    if k_iter % 5 == 0
        println("  it=$k_iter  loss=$(round(L; digits=6))"); flush(stdout)
    end
    return false
end

initial_lr = 0.0010
decay_lr   = 0.00035
switch_it  = 40

function train_with_schedule()
    it1 = min(MAXITERS, switch_it)
    opt1 = Optimisers.OptimiserChain(Optimisers.ClipNorm(1.0f0), Optimisers.Adam(initial_lr))
    stg1 = solve(optprob, opt1; maxiters=it1, callback=cb)
    rem = MAXITERS - it1
    if rem > 0
        optprob2 = OptimizationProblem(optf, ComponentArray(stg1.u))
        opt2 = Optimisers.OptimiserChain(Optimisers.ClipNorm(1.0f0), Optimisers.Adam(decay_lr))
        stg2 = solve(optprob2, opt2; maxiters=rem, callback=cb)
        return stg2
    else
        return stg1
    end
end

solopt = train_with_schedule()

θ★ = ComponentArray(solopt.u)
```

### 6. Evaluation and Visualization
Finally, after training, we evaluate the performance of our UDE. We solve the UDE one last time using the final, optimized parameters (θ★). We then create two plots to compare the UDE's solution to the ground truth:

A heatmap showing the spatial concentration of one species at the final time point.

A time-series plot showing the evolution of the mean concentration over the entire simulation.

If the training was successful, the UDE's output should closely match the true simulation.

```@example stiff_bruss
println("\nStage 4/4: Final evaluation …"); flush(stdout)
println("  Config: N=$(N), TEND=$(TEND), TRAIN_FRAC=$(TRAIN_FRAC)")
println("  Threads during final solve: DISABLED | Algorithm: ", typeof(ALG_FINAL))
t_start = time()

sol_ude_final = solve(remake(prob_ude; p=θ★), ALG_FINAL;
          saveat=ts, reltol=FINAL_RTOL, abstol=FINAL_ATOL,
          save_everystep=false, dtmax=FINAL_DTMAX)

println("  Final solve retcode = ", sol_ude_final.retcode)
println("  Saved points = ", length(sol_ude_final.t))
println("  Final solve wall time = $(round(time()-t_start; digits=3)) s")

rel_mse = sum(abs2, Array(sol_ude_final) .- Yref) / sum(abs2, Yref)
println("done. Final relative MSE = ", rel_mse)

mean_ude = [mean(u[:,:,1]) for u in sol_ude_final.u]
plt_final = plot(ts, mean_true, label="True Simulation", lw=2,
                 xlabel="Time (t)", ylabel="Mean Concentration",
                 title="Model Performance (Final)")
plot!(plt_final, ts, mean_ude, label=RESIDUAL ? "SNN Residual" : "SNN-UDE Prediction",
      lw=2, linestyle=:dash)
```
