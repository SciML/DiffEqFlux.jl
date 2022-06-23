# DiffEqFlux: High Level Pre-Built Architectures for Implicit Deep Learning

DiffEqFlux.jl is an implicit deep learning library built using the SciML ecosystem. It is
a high level interface that pulls together all of the tools with heuristics
and helper functions to make training such deep implicit layer models fast and easy.

!!! note
    
    DiffEqFlux.jl is only for pre-built architectures and utility functions
    for deep implicit learning, mixing differential equations with machine
    learning. For details on automatic differentiation of equation solvers
    and adjoint techniques, and using these methods for doing things like
    callibrating models to data, nonlinear optimal control, and PDE-constrained
    optimization, see [SciMLSensitivity.jl](https://sensitivity.sciml.ai/dev/)

## Pre-Built Architectures

The approach of this package is the easy and efficient training of
[Universal Differential Equations](https://arxiv.org/abs/2001.04385).
DiffEqFlux.jl provides architectures which match the interfaces of
machine learning libraries such as [Flux.jl](https://fluxml.ai/)
and [Lux.jl](http://lux.csail.mit.edu/dev/)
to make it easy to build continuous-time machine learning layers
into larger machine learning applications.

The following layer functions exist:

  - [Neural Ordinary Differential Equations (Neural ODEs)](https://arxiv.org/abs/1806.07366)
  - [Collocation-Based Neural ODEs (Neural ODEs without a solver, by far the fastest way!)](https://www.degruyter.com/document/doi/10.1515/sagmb-2020-0025/html)
  - [Multiple Shooting Neural Ordinary Differential Equations](https://arxiv.org/abs/2109.06786)
  - [Neural Stochastic Differential Equations (Neural SDEs)](https://arxiv.org/abs/1907.07587)
  - [Neural Differential-Algebriac Equations (Neural DAEs)](https://arxiv.org/abs/2001.04385)
  - [Neural Delay Differential Equations (Neural DDEs)](https://arxiv.org/abs/2001.04385)
  - [Augmented Neural ODEs](https://arxiv.org/abs/1904.01681)
  - [Hamiltonian Neural Networks (with specialized second order and symplectic integrators)](https://arxiv.org/abs/1906.01563)
  - [Continuous Normalizing Flows (CNF)](https://arxiv.org/abs/1806.07366) and [FFJORD](https://arxiv.org/abs/1810.01367)

Examples of how to build architectures from scratch, with tutorials on things
like Graph Neural ODEs, can be found in the [SciMLSensitivity.jl documentation](sensitivity.sciml.ai/dev).

WIP:

  - Lagrangian Neural Networks
  - Galerkin Neural ODEs

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
