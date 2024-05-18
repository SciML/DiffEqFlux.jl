# DiffEqFlux.jl

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/DiffEqFlux/stable/)

[![codecov](https://codecov.io/gh/SciML/DiffEqFlux.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/SciML/DiffEqFlux.jl)
[![Build Status](https://github.com/SciML/DiffEqFlux.jl/workflows/CI/badge.svg)](https://github.com/SciML/DiffEqFlux.jl/actions?query=workflow%3ACI)
[![Build status](https://badge.buildkite.com/a1fecf87b085b452fe0f3d3968ddacb5c1d5570806834e1d52.svg)](https://buildkite.com/julialang/diffeqflux-dot-jl)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

DiffEq(For)Lux.jl (aka DiffEqFlux.jl) fuses the world of differential equations with machine learning
by helping users put diffeq solvers into neural networks. This package utilizes
[DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/), and [Lux.jl](https://lux.csail.mit.edu/)  as its building blocks to support research in
[Scientific Machine Learning](https://www.stochasticlifestyle.com/the-essential-tools-of-scientific-machine-learning-scientific-ml/), specifically neural differential equations to add physical information into traditional machine learning.

> [!NOTE]
> We maintain backwards compatibility with [Flux.jl](https://docs.sciml.ai/Flux/stable/) via [FromFluxAdaptor()](https://lux.csail.mit.edu/stable/api/Lux/flux_to_lux#FromFluxAdaptor())

## Tutorials and Documentation

For information on using the package,
[see the stable documentation](https://docs.sciml.ai/DiffEqFlux/stable/). Use the
[in-development documentation](https://docs.sciml.ai/DiffEqFlux/dev/) for the version of
the documentation, which contains the unreleased features.

## Problem Domain

DiffEqFlux.jl is for implicit layer machine learning.
DiffEqFlux.jl provides architectures which match the interfaces of machine learning libraries such as Flux.jl and Lux.jl to make it easy to build continuous-time machine learning layers into larger machine learning applications.

The following layer functions exist:

  - Neural Ordinary Differential Equations (Neural ODEs)
  - Collocation-Based Neural ODEs (Neural ODEs without a solver, by far the fastest way!)
  - Multiple Shooting Neural Ordinary Differential Equations
  - Neural Stochastic Differential Equations (Neural SDEs)
  - Neural Differential-Algebraic Equations (Neural DAEs)
  - Neural Delay Differential Equations (Neural DDEs)
  - Augmented Neural ODEs
  - Hamiltonian Neural Networks (with specialized second order and symplectic integrators)
  - Continuous Normalizing Flows (CNF) and FFJORD

with high order, adaptive, implicit, GPU-accelerated, Newton-Krylov, etc.
methods. For examples, please refer to
[the release blog post](https://julialang.org/blog/2019/01/fluxdiffeq).
Additional demonstrations, like neural
PDEs and neural jump SDEs, can be found
[in this blog post](https://www.stochasticlifestyle.com/neural-jump-sdes-jump-diffusions-and-neural-pdes/)
(among many others!).

Do not limit yourself to the current neuralization. With this package, you can
explore various ways to integrate the two methodologies:

  - Neural networks can be defined where the “activations” are nonlinear functions
    described by differential equations
  - Neural networks can be defined where some layers are ODE solves
  - ODEs can be defined where some terms are neural networks
  - Cost functions on ODEs can define neural networks

![Flux ODE Training Animation](https://user-images.githubusercontent.com/1814174/88589293-e8207f80-d026-11ea-86e2-8a3feb8252ca.gif)

## Breaking Changes in v3

  - Flux dependency is dropped. If a non Lux `AbstractExplicitLayer` is passed we try to automatically convert it to a Lux model with `FromFluxAdaptor()(model)`.
  - `Flux` is no longer re-exported from `DiffEqFlux`. Instead we reexport `Lux`.
  - `NeuralDAE` now allows an optional `du0` as input.
  - `TensorLayer` is now a Lux Neural Network.
  - APIs for quite a few layer constructions have changed. Please refer to the updated documentation for more details.
