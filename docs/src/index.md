# DiffEqFlux: High Level Pre-Built Architectures for Implicit Deep Learning

DiffEqFlux.jl is an implicit deep learning library built using the SciML ecosystem. It is
a high-level interface that pulls together all the tools with heuristics
and helper functions to make training such deep implicit layer models fast and easy.

!!! note
    
    DiffEqFlux.jl is only for pre-built architectures and utility functions
    for deep implicit learning, mixing differential equations with machine
    learning. For details on automatic differentiation of equation solvers
    and adjoint techniques, and using these methods for doing things like
    calibrating models to data, nonlinear optimal control, and PDE-constrained
    optimization, see [SciMLSensitivity.jl](https://docs.sciml.ai/SciMLSensitivity/stable/).

## Pre-Built Architectures

The approach of this package is the easy and efficient training of
[Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366) and its variants.
DiffEqFlux.jl provides architectures which match the interfaces of
machine learning libraries such as [Flux.jl](https://docs.sciml.ai/Flux/stable/)
and [Lux.jl](https://lux.csail.mit.edu/stable/api/)
to make it easy to build continuous-time machine learning layers
into larger machine learning applications.

The following layer functions exist:

  - [Neural Ordinary Differential Equations (Neural ODEs)](https://arxiv.org/abs/1806.07366)
  - [Collocation-Based Neural ODEs (Neural ODEs without a solver, by far the fastest way!)](https://www.degruyter.com/document/doi/10.1515/sagmb-2020-0025/html)
  - [Multiple Shooting Neural Ordinary Differential Equations](https://arxiv.org/abs/2109.06786)
  - [Neural Stochastic Differential Equations (Neural SDEs)](https://arxiv.org/abs/1907.07587)
  - [Neural Differential-Algebraic Equations (Neural DAEs)](https://arxiv.org/abs/2001.04385)
  - [Neural Delay Differential Equations (Neural DDEs)](https://arxiv.org/abs/2001.04385)
  - [Augmented Neural ODEs](https://arxiv.org/abs/1904.01681)
  - [Hamiltonian Neural Networks (with specialized second order and symplectic integrators)](https://arxiv.org/abs/1906.01563)
  - [Continuous Normalizing Flows (CNF)](https://arxiv.org/abs/1806.07366) and [FFJORD](https://arxiv.org/abs/1810.01367)

Examples of how to build architectures from scratch, with tutorials on things
like Graph Neural ODEs, can be found in the [SciMLSensitivity.jl documentation](https://docs.sciml.ai/SciMLSensitivity/stable/).

## Flux.jl vs Lux.jl

Both Flux and Lux defined neural networks are supported by DiffEqFlux.jl. However, Lux.jl neural networks are greatly preferred for many
correctness reasons. Particularly, a Flux `Chain` does not respect Julia's type promotion rules. This causes major problems in that
the restructuring of a Flux neural network will not respect the chosen types from the solver. Demonstration:

```julia
using Flux, Tracker

x = [0.8; 0.8]
ann = Chain(Dense(2, 10, tanh), Dense(10, 1))
p, re = Flux.destructure(ann)
z = re(Float64.(p))
```

While one may think this recreates the neural network to act in `Float64` precision, [it does not](https://github.com/FluxML/Flux.jl/pull/2156)
and instead its values will silently downgrade everything to `Float32`. This is only fixed by `Chain(Dense(2, 10, tanh), Dense(10, 1)) |> f64`.
Similar cases will [lead to dropped gradients with complex numbers](https://github.com/FluxML/Optimisers.jl/issues/95). This is not an issue
with the automatic differentiation library commonly associated with Flux (Zygote.jl) but rather due to choices in the neural network library's
decision for how to approach type handling and precision. Thus when using DiffEqFlux.jl with Flux, the user must be very careful to ensure that
the precision of the arguments are correct, and anything that requires alternative types (like `TrackerAdjoint` tracked values and
`ForwardDiffSensitivity` dual numbers) are suspect.

Lux.jl has none of these issues, is simpler to work with due to the parameters in its function calls being explicit rather than implicit global
references, and achieves higher performance. It is built on the same foundations as Flux.jl, such as Zygote and NNLib, and thus it supports the
same layers underneath and calls the same kernels. The better performance comes from not having the overhead of `restructure` required.
Thus we highly recommend people use Lux instead and only use the Flux fallbacks for legacy code.

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

## Reproducibility

```@raw html
<details><summary>The documentation of this SciML package was built using these direct dependencies,</summary>
```

```@example
using Pkg # hide
Pkg.status() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>and using this machine and Julia version.</summary>
```

```@example
using InteractiveUtils # hide
versioninfo() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>
```

```@example
using Pkg # hide
Pkg.status(; mode = PKGMODE_MANIFEST) # hide
```

```@raw html
</details>
```

```@eval
using TOML
using Markdown
version = TOML.parse(read("../../Project.toml", String))["version"]
name = TOML.parse(read("../../Project.toml", String))["name"]
link_manifest = "https://github.com/SciML/" *
                name *
                ".jl/tree/gh-pages/v" *
                version *
                "/assets/Manifest.toml"
link_project = "https://github.com/SciML/" *
               name *
               ".jl/tree/gh-pages/v" *
               version *
               "/assets/Project.toml"
Markdown.parse("""You can also download the
[manifest]($link_manifest)
file and the
[project]($link_project)
file.
""")
```
