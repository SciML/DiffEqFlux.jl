# GalacticOptim.jl

*Important:* Please note that `sciml_train` has now been replaced by
[GalacticOptim.jl](https://github.com/SciML/GalacticOptim.jl). The conversion to
GalacitcOptim.jl-style should not pose too many problems! Consult
[this](https://diffeqflux.sciml.ai/dev/examples/neural_ode_sciml/) tutorial
to see how an optimization problem can be set up and/or also read the updated
[GalacticOptim.jl documentation](https://galacticoptim.sciml.ai/stable/)
to explore more options in detail.

GalacticOptim.jl is a package with a scope that is beyond your normal global optimization
package. GalacticOptim.jl seeks to bring together all of the optimization packages
it can find, local and global, into one unified Julia interface. This means, you
learn one package and you learn them all! GalacticOptim.jl adds a few high-level
features, such as integrating with automatic differentiation, to make its usage
fairly simple for most cases, while allowing all of the options in a single
unified interface.

##### Note: This package is still in active development.

## Installation

Assuming that you already have Julia correctly installed, it suffices to import
GalacticOptim.jl in the standard way:

```julia
import Pkg; Pkg.add("GalacticOptim")
```
The packages relevant to the core functionality of GalacticOptim.jl will be imported
accordingly and, in most cases, you do not have to worry about the manual
installation of dependencies. Below is the list of packages that need to be
installed explicitly if you intend to use the specific optimization algorithms
offered by them:

- [BlackBoxOptim.jl](https://github.com/robertfeldt/BlackBoxOptim.jl) (solver: `BBO()`)
- [NLopt.jl](https://github.com/JuliaOpt/NLopt.jl) (usage via the NLopt API;
see also the available [algorithms](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/))
- [MultistartOptimization.jl](https://github.com/tpapp/MultistartOptimization.jl)
(see also [this documentation](https://juliahub.com/docs/MultistartOptimization/cVZvi/0.1.0/))
- [QuadDIRECT.jl](https://github.com/timholy/QuadDIRECT.jl)
- [Evolutionary.jl](https://github.com/wildart/Evolutionary.jl) (see also [this documentation](https://wildart.github.io/Evolutionary.jl/dev/))
- [CMAEvolutionStrategy.jl](https://github.com/jbrea/CMAEvolutionStrategy.jl)

## Tutorials and Documentation

For information on using the package,
[see the stable documentation](https://galacticoptim.sciml.ai/stable/). Use the
[in-development documentation](https://galacticoptim.sciml.ai/dev/) for the version of
the documentation, which contains the unreleased features.

## Examples

```julia
 using GalacticOptim, Optim
 rosenbrock(x,p) =  (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
 x0 = zeros(2)
 p  = [1.0,100.0]

 prob = OptimizationProblem(rosenbrock,x0,p)
 sol = solve(prob,NelderMead())


 using BlackBoxOptim
 prob = OptimizationProblem(rosenbrock, x0, p, lb = [-1.0,-1.0], ub = [1.0,1.0])
 sol = solve(prob,BBO())
```

Note that Optim.jl is a core dependency of GalaticOptim.jl. However, BlackBoxOptim.jl
is not and must already be installed (see the list above).

The output of the first optimization task (with the `NelderMead()` algorithm)
is given below:

```julia
* Status: success

* Candidate solution
   Final objective value:     3.525527e-09

* Found with
   Algorithm:     Nelder-Mead

* Convergence measures
   √(Σ(yᵢ-ȳ)²)/n ≤ 1.0e-08

* Work counters
   Seconds run:   0  (vs limit Inf)
   Iterations:    60
   f(x) calls:    118
```
We can also explore other methods in a similar way:

```julia
 f = OptimizationFunction(rosenbrock, GalacticOptim.AutoForwardDiff())
 prob = OptimizationProblem(f, x0, p)
 sol = solve(prob,BFGS())
```
For instance, the above optimization task produces the following output:

```julia
* Status: success

* Candidate solution
   Final objective value:     7.645684e-21

* Found with
   Algorithm:     BFGS

* Convergence measures
   |x - x'|               = 3.48e-07 ≰ 0.0e+00
   |x - x'|/|x'|          = 3.48e-07 ≰ 0.0e+00
   |f(x) - f(x')|         = 6.91e-14 ≰ 0.0e+00
   |f(x) - f(x')|/|f(x')| = 9.03e+06 ≰ 0.0e+00
   |g(x)|                 = 2.32e-09 ≤ 1.0e-08

* Work counters
   Seconds run:   0  (vs limit Inf)
   Iterations:    16
   f(x) calls:    53
   ∇f(x) calls:   53
```

```julia
 prob = OptimizationProblem(f, x0, p, lb = [-1.0,-1.0], ub = [1.0,1.0])
 sol = solve(prob, Fminbox(GradientDescent()))
```
The examples clearly demonstrate that GalacticOptim.jl provides an intuitive
way of specifying optimization tasks and offers a relatively
easy access to a wide range of optimization algorithms.
