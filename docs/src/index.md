# DiffEqFlux: Generalized Physics-Informed and Scientific Machine Learning (SciML)

DiffEqFlux.jl is not just for neural ordinary differential equations.
DiffEqFlux.jl is for universal differential equations, where these can include
delays, physical constraints, stochasticity, events, and all other kinds of
interesting behavior that shows up in scientific simulations. Neural networks can
be all or part of the model. They can be around the differential equation,
in the cost function, or inside of the differential equation. Neural networks
representing unknown portions of the model or functions can go anywhere you
have uncertainty in the form of the scientific simulator. For an overview of the
topic with applications, consult the paper [Universal Differential Equations for
Scientific Machine Learning](https://arxiv.org/abs/2001.04385).

As such, it is the first package to support and demonstrate:

- Stiff universal ordinary differential equations (universal ODEs)
- Universal stochastic differential equations (universal SDEs)
- Universal delay differential equations (universal DDEs)
- Universal partial differential equations (universal PDEs)
- Universal jump stochastic differential equations (universal jump diffusions)
- Hybrid universal differential equations (universal DEs with event handling)

with high order, adaptive, implicit, GPU-accelerated, Newton-Krylov, etc.
methods. For examples, please refer to [the release blog
post](https://julialang.org/blog/2019/01/fluxdiffeq) (which we try to keep
updated for changes to the libraries). Additional demonstrations, like neural
PDEs and neural jump SDEs, can be found [at this blog
post](http://www.stochasticlifestyle.com/neural-jump-sdes-jump-diffusions-and-neural-pdes/)
(among many others!).

Many different training techniques are supported by this package, including:

- Optimize-then-discretize (backsolve adjoints, checkpointed adjoints, quadrature adjoints)
- Discretize-then-optimize (forward and reverse mode discrete sensitivity analysis)
  - This is a generalization of [ANODE](https://arxiv.org/pdf/1902.10298.pdf) and [ANODEv2](https://arxiv.org/pdf/1906.04596.pdf) to all [DifferentialEquations.jl ODE solvers](https://diffeq.sciml.ai/latest/solvers/ode_solve/)
- Hybrid approaches (adaptive time stepping + AD for adaptive discretize-then-optimize)
- Collocation approaches (two-stage methods, multiple shooting, etc.)

For more details on the adjoint sensitivity analysis methods for computing
fast gradients, see the [Adjoints page](@ref adjoints).

With this package, you can explore various ways to integrate the two methodologies:

- Neural networks can be defined where the “activations” are nonlinear functions
  described by differential equations
- Neural networks can be defined where some layers are ODE solves
- ODEs can be defined where some terms are neural networks
- Cost functions on ODEs can define neural networks

## Basics

The basics are all provided by the
[DifferentialEquations.jl](https://diffeq.sciml.ai/latest/) package. Specifically,
[the `solve` function is automatically compatible with AD systems like Zygote.jl](https://diffeq.sciml.ai/latest/analysis/sensitivity/)
and thus there is no machinery that is necessary to use DifferentialEquations.jl
package. For example, the following computes the solution to an ODE and computes
the gradient of a loss function (the sum of the ODE's output at each timepoint
with dt=0.1) via the adjoint method:

```julia
using DiffEqSensitivity, OrdinaryDiffEq, Zygote

function fiip(du,u,p,t)
  du[1] = dx = p[1]*u[1] - p[2]*u[1]*u[2]
  du[2] = dy = -p[3]*u[2] + p[4]*u[1]*u[2]
end
p = [1.5,1.0,3.0,1.0]; u0 = [1.0;1.0]
prob = ODEProblem(fiip,u0,(0.0,10.0),p)
sol = solve(prob,Tsit5())
loss(u0,p) = sum(solve(prob,Tsit5(),u0=u0,p=p,saveat=0.1))
du01,dp1 = Zygote.gradient(loss,u0,p)
```

Thus, what DiffEqFlux.jl provides is:

- A bunch of tutorials, documentation, and test cases for this combination
  with neural network libraries and GPUs
- Pre-built layer functions for common use cases, like neural ODEs
- Specialized layer functions (`FastDense`) to improve neural differential equation
  training performance
- A specialized optimization function `sciml_train` with a training loop that
  allows non-machine learning libraries to be easily utilized

## Applications

The approach of this package is the efficient training of
[Universal Differential Equations](https://arxiv.org/abs/2001.04385).
Since this is a fairly general class of problems, the following
applications are readily available as specific instances of this
methodology, and are showcased in tutorials and layer functions:

- Neural ODEs
- Neural SDEs
- Neural DAEs
- Neural DDEs
- Augmented Neural ODEs
- Graph Neural ODEs
- Hamiltonian Neural Networks (with specialized second order and symplectic integrators)
- Legrangian Neural Networks
- Continuous Normalizing Flows (CNF) and FFJORD
- Galerkin Nerual ODEs

## Citation

If you use DiffEqFlux.jl or are influenced by its ideas, please cite:

```
@article{rackauckas2020universal,
  title={Universal differential equations for scientific machine learning},
  author={Rackauckas, Christopher and Ma, Yingbo and Martensen, Julius and Warner, Collin and Zubov, Kirill and Supekar, Rohit and Skinner, Dominic and Ramadhan, Ali},
  journal={arXiv preprint arXiv:2001.04385},
  year={2020}
}
```
