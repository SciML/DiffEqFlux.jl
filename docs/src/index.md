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
(among many others!). All of these features are only part of the advantage, as this library
[routinely benchmarks orders of magnitude faster than competing libraries like torchdiffeq](@ref Benchmarks)

Many different training techniques are supported by this package, including:

- Optimize-then-discretize (backsolve adjoints, checkpointed adjoints, quadrature adjoints)
- Discretize-then-optimize (forward and reverse mode discrete sensitivity analysis)
  - This is a generalization of [ANODE](https://arxiv.org/pdf/1902.10298.pdf) and [ANODEv2](https://arxiv.org/pdf/1906.04596.pdf) to all [DifferentialEquations.jl ODE solvers](https://diffeq.sciml.ai/latest/solvers/ode_solve/)
- Hybrid approaches (adaptive time stepping + AD for adaptive discretize-then-optimize)
- Collocation approaches (two-stage methods, multiple shooting, etc.)
- O(1) memory backprop of ODEs via BacksolveAdjoint, and Virtual Brownian Trees for O(1) backprop of SDEs
- [Continuous adjoints for integral loss functions](https://diffeq.sciml.ai/stable/analysis/sensitivity/#Example-continuous-adjoints-on-an-energy-functional)
- Probabilistic programming and variational inference on ODEs/SDEs/DAEs/DDEs/hybrid
  equations etc. is provided by integration with [Turing.jl](https://turing.ml/dev/)
  and [Gen.jl](https://github.com/probcomp/Gen.jl). Reproduce
  [variational loss functions](https://arxiv.org/abs/2001.01328) by plugging
  [composible libraries together](https://turing.ml/dev/tutorials/9-variationalinference/).

all while mixing forward mode and reverse mode approaches as appropriate for the
most speed. For more details on the adjoint sensitivity analysis methods for
computing fast gradients, see the [Adjoints page](@ref adjoints).

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

## Modularity and Composability

Note that DiffEqFlux.jl purely built on composable and modular infrustructure. In fact, 
DiffEqFlux.jl's functions are not even directly required for performing many of these operations! 
DiffEqFlux provides high level helper functions and documentation for the user, but the 
code generation stack is modular and composes in many different ways. For example, one can 
use and swap out the ODE solver between any common interface compatible library, like:

- Sundials.jl
- OrdinaryDiffEq.jl
- LSODA.jl
- [IRKGaussLegendre.jl](https://github.com/mikelehu/IRKGaussLegendre.jl)
- [SciPyDiffEq.jl](https://github.com/SciML/SciPyDiffEq.jl)
- [... etc. many other choices!](https://diffeq.sciml.ai/stable/solvers/ode_solve/)

In addition, due to the composability of the system, none of the components are directly
tied to the Flux.jl machine learning framework. For example, you can [use DiffEqFlux.jl
to generate TensorFlow graphs and train the nueral network with TensorFlow.jl](https://youtu.be/n2MwJ1guGVQ?t=284),
[utilize PyTorch arrays via Torch.jl](https://github.com/FluxML/Torch.jl), and more all with
single line code changes by utilizing the underlying code generation. The tutorials shown here
are thus mostly a guide on how to use the ecosystem as a whole, only showing a small snippet
of the possible ways to compose the thousands of differentiable libraries together! Swap out
ODEs for SDEs, DDEs, DAEs, etc., put quadrature libraries or [Tullio.jl](https://github.com/mcabbott/Tullio.jl)
in the loss function, the world is your oyster!

As a proof of composability, note that the implementation of Bayesian neural ODEs required
zero code changes to the library, and instead just relied on the composability with other
Julia packages.

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
