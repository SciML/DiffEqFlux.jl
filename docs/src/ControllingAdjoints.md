# [Controlling Choices of Adjoints](@id adjoints)

DiffEqFlux is capable of training neural networks embedded inside of
differential equations with many different techniques. For all of the
details, see the
[DifferentialEquations.jl local sensitivity analysis](https://diffeq.sciml.ai/latest/analysis/sensitivity/)
documentation. Here we will summarize these methodologies in the
context of neural differential equations and scientific machine learning.

## Choosing a sensealg in a Nutshell

By default, a stable adjoint with an auto-adapting vjp choice is used. In many cases, a user can optimize the choice to compute more than an order of magnitude faster than the default. However, given the vast space to explore, use the following decision tree to help guide the choice:

- If you have 100 parameters or less, consider using forward-mode sensititivites. If the `f` function is not ForwardDiff-compatible, use `ForwardSensitivty`, otherwise use `ForwardDiffSensitivty` as its more efficient.
- For larger equations, give `BacksolveAdjoint` and `InterpolatingAdjoint` a try. If the gradient of `BacksolveAdjoint` is correct, many times it's the faster choice so choose that (but it's not always faster!). If your equation is stiff or a DAE, skip this step as `BacksolveAdjoint` is almost certainly unstable.
- If your equation does not use much memory and you're using a stiff solver, consider using `QuadratureAdjoint` as it is asymtopically more computationally efficient by trading off memory cost.
- If the other methods are all unstable (check the gradients against each other!), then `ReverseDiffAdjoint` is a good fallback on CPU, while `TrackerAdjoint` is a good fallback on GPUs.
- After choosing a general sensealg, if the choice is `InterpolatingAdjoint`, `QuadratureAdjoint`, or `BacksolveAdjoint`, then optimize the choice of vjp calculation next:
  - If your function has no branching (no if statements), use `ReverseDiffVJP(true)`.
  - If you're on the CPU and your function is very scalarized in operations but has branches, choose `ReverseDiffVJP()`.
  - If your on the CPU or GPU and your function is very vectorized, choose `ZygoteVJP()`.
  - Else fallback to `TrackerVJP()` if Zygote does not support the function.

## Additional Details

A sensitivity analysis method can be passed to a solver via the `sensealg`
keyword argument. For example:

```julia
solve(prob,Tsit5(),sensealg=BacksolveAdjoint(autojacvec=ZygoteVJP()))
```

sets the adjoint sensitivity analysis so that, when this call is
encountered in the gradient calculation of any of the Julia reverse-mode
AD frameworks, the differentiation will be replaced with the `BacksolveAdjoint`
method where internal vector-Jacobian products are performed using
Zygote.jl. From the [DifferentialEquations.jl local sensitivity analysis](https://diffeq.sciml.ai/latest/analysis/sensitivity/)
page, we note that the following choices for `sensealg` exist:

- `BacksolveAdjoint`
- `InterpolatingAdjoint` (with checkpoints)
- `QuadratureAdjoint`
- `TrackerAdjoint`
- `ReverseDiffAdjoint` (currently requires `using DistributionsAD`)
- `ZygoteAdjoint` (currently limited to special solvers)

Additionally, there are methodologies for forward sensitivity analysis:

- `ForwardSensitivty`
- `ForwardDiffSensitivty`

These methods have very low overhead compared to adjoint methods but
have poor scaling with respect to increased numbers of parameters.
[Our benchmarks demonstrate a cutoff of around 100 parameters](https://arxiv.org/abs/1812.01892),
where for models with less than 100 parameters these techniques are more
efficient, but when there are more than 100 parameters (like in neural ODEs)
these methods are less efficient than the adjoint methods.

## Choices of Vector-Jacobian Products (autojacvec)

With each of these solvers, `autojacvec` can be utilized to choose how
the internal vector-Jacobian products of the `f` function are computed.
The choices are:

- `ReverseDiffVJP(compile::Bool)`: Usually the fastest when scalarized operations exist in the `f` function (like
  in scientific machine learning applications like Universal Differential
  Equations) and the boolean `compile` (i.e. `ReverseDiffVJP(true)`)
  is the absolute fastest but requires that the `f` function of the
  ODE/DAE/SDE/DDE has no branching. Does not support GPUs. 

- `TrackerVJP`: Not as efficient as `ReverseDiffVJP`, but supports GPUs. 

- `ZygoteVJP`: Tends to be the fastest VJP method if the ODE/DAE/SDE/DDE is written with mostly vectorized functions (like neural networks and
  other layers from [Flux.jl](https://fluxml.ai/)). Bear in mind that Zygote does not allow mutation, making the solve more memory expensive and therefore slow.

- `nothing`: Default choice given characteristics of the types in your model.
- `true`: Forward-mode AD Jacobian-vector products. Should only be used on sufficiently small equations

- `false`: Numerical Jacobian-vector products. Should only be used if the `f` function is not differentiable
  (i.e. is a Fortran code).

As other vector-Jacobian product systems become available
in Julia they will be added to this system so that no user code changes
are required to interface with these methodologies. 

## Manual VJPs

Note that when defining your differential equation the vjp can be
manually overwritten by providing a `vjp(u,p,t)` that returns a tuple
`f(u,p,t),v->J*v` in the form of [ChainRules.jl](https://www.juliadiff.org/ChainRulesCore.jl/stable/).
When this is done, the choice of `ZygoteVJP` will utilize your VJP
function during the internal steps of the adjoint. This is useful for
models where automatic differentiation may have trouble producing
optimal code. This can be paired with [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl)
for producing hyper-optimized, sparse, and parallel VJP functions utilizing
the automated symbolic conversions.

## Optimize-then-Discretize

[The original neural ODE paper](https://arxiv.org/abs/1806.07366)
popularized optimize-then-discretize with O(1) adjoints via backsolve.
This is the methodology `BacksolveAdjoint`
When training non-stiff neural ODEs, `BacksolveAdjoint` with `ZygoteVJP`
is generally the fastest method. Additionally, this method does not
require storing the values of any intermediate points and is thus the
most memory efficient. However, `BacksolveAdjoint` is prone
to instabilities whenever the Lipschitz constant is sufficiently large,
like in stiff equations, PDE discretizations, and many other contexts,
so it is not used by default. When training a neural ODE for machine
learning applications, the user should try `BacksolveAdjoint` and see
if it is sufficiently accurate on their problem.

Note that DiffEqFlux's implementation of `BacksolveAdjoint` includes
an extra feature `BacksolveAdjoint(checkpointing=true)` which mixes
checkpointing with `BacksolveAdjoint`. What this method does is that,
at `saveat` points, values from the forward pass are saved. Since the
reverse solve should numerically be the same as the forward pass, issues
with divergence of the reverse pass are mitigated by restarting the
reverse pass at the `saveat` value from the forward pass. This reduces
the divergence and can lead to better gradients at the cost of higher
memory usage due to having to save some values of the forward pass.
This can stabilize the adjoint in some applications, but for highly
stiff applications the divergence can be too fast for this to work in
practice.

To avoid the issues of backwards solving the ODE, `InterpolatingAdjoint`
and `QuadratureAdjoint` utilize information from the forward pass.
By default these methods utilize the [continuous solution](https://diffeq.sciml.ai/latest/basics/solution/#Interpolations-1)
provided by DifferentialEquations.jl in the calculations of the
adjoint pass. `QuadratureAdjoint` uses this to build a continuous
function for the solution of adjoint equation and then performs an
adaptive quadrature via [Quadrature.jl](https://github.com/SciML/Quadrature.jl),
while `InterpolatingAdjoint` appends the integrand to the ODE so it's
computed simultaneously to the Lagrange multiplier. When memory is
not an issue, we find that the `QuadratureAdjoint` approach tends to
be the most efficient as it has a significantly smaller adjoint
differential equation and the quadrature converges very fast, but this
form requires holding the full continuous solution of the adjoint which
can be a significant burden for large parameter problems. The
`InterpolatingAdjoint` is thus a compromise between memory efficiency
and compute efficiency, and is in the same spirit as [CVODES](https://computing.llnl.gov/projects/sundials).

However, if the memory cost of the `InterpolatingAdjoint` is too high,
checkpointing can be used via `InterpolatingAdjoint(checkpointing=true)`.
When this is used, the checkpoints default to `sol.t` of the forward
pass (i.e. the saved timepoints usually set by `saveat`). Then in the
adjoint, intervals of `sol.t[i-1]` to `sol.t[i]` are re-solved in order
to obtain a short interpolation which can be utilized in the adjoints.
This at most results in two full solves of the forward pass, but
dramatically reduces the computational cost while being a low-memory
format. This is the preferred method for highly stiff equations
when memory is an issue, i.e. stiff PDEs or large neural DAEs.

For forward-mode, the `ForwardSensitivty` is the version that performs
the optimize-then-discretize approach. In this case, `autojacvec` corresponds
to the method for computing `J*v` within the forward sensitivity equations,
which is either `true` or `false` for whether to use Jacobian-free
forward-mode AD (via ForwardDiff.jl) or Jacobian-free numerical
differentiation.

## Discretize-then-Optimize

In this approach the discretization is done first and then optimization
is done on the discretized system. While traditionally this can be
done discrete sensitivity analysis, this is can be equivalently done
by automatic differentiation on the solver itself. `ReverseDiffAdjoint`
performs reverse-mode automatic differentiation on the solver via
[ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl),
`ZygoteAdjoint` performs reverse-mode automatic
differentiation on the solver via
[Zygote.jl](https://github.com/FluxML/Zygote.jl), and `TrackerAdjoint`
performs reverse-mode automatic differentiation on the solver via
[Tracker.jl](https://github.com/FluxML/Tracker.jl). In addition,
`ForwardDiffSensitivty` performs forward-mode automatic differentiation
on the solver via [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl).

We note that many studies have suggested that [this approach produces
more accurate gradients than the optimize-than-discretize approach](https://arxiv.org/abs/2005.13420)

# Special Notes on Equation Types

While all of the choices are compatible with ordinary differential
equations, specific notices apply to other forms:

## Differential-Algebraic Equations

We note that while all 3 are compatible with index-1 DAEs via the
[derivation in the universal differential equations paper](https://arxiv.org/abs/2001.04385)
(note the reinitialization), we do not recommend `BacksolveAdjoint`
one DAEs because the stiffness inherent in these problems tends to
cause major difficulties with the accuracy of the backwards solution
due to reinitialization of the algebraic variables.

## Stochastic Differential Equations

We note that all of the adjoints except `QuadratureAdjoint` are applicable
to stochastic differential equations.

## Delay Differential Equations

We note that only the discretize-then-optimize methods are applicable
to delay differential equations. Constant lag and variable lag
delay differential equation parameters can be estimated, but the lag
times themselves are unable to be estimated through these automatic
differentiation techniques.
