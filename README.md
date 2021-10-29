# DiffEqFlux.jl

[![Join the chat at https://gitter.im/JuliaDiffEq/Lobby](https://badges.gitter.im/JuliaDiffEq/Lobby.svg)](https://gitter.im/JuliaDiffEq/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://github.com/SciML/DiffEqFlux.jl/workflows/CI/badge.svg)](https://github.com/SciML/DiffEqFlux.jl/actions?query=workflow%3ACI)
[![Build status](https://badge.buildkite.com/a1fecf87b085b452fe0f3d3968ddacb5c1d5570806834e1d52.svg)](https://buildkite.com/julialang/diffeqflux-dot-jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](http://diffeqflux.sciml.ai/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](http://diffeqflux.sciml.ai/dev/)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)


DiffEqFlux.jl fuses the world of differential equations with machine learning
by helping users put diffeq solvers into neural networks. This package utilizes
[DifferentialEquations.jl](http://diffeq.sciml.ai/dev/) and
[Flux.jl](https://fluxml.ai/) as its building blocks to support research in
[Scientific Machine Learning](http://www.stochasticlifestyle.com/the-essential-tools-of-scientific-machine-learning-scientific-ml/),
specifically neural differential equations and universal differential equations,
to add physical information into traditional machine learning.

## Tutorials and Documentation

For information on using the package,
[see the stable documentation](https://diffeqflux.sciml.ai/stable/). Use the
[in-development documentation](https://diffeqflux.sciml.ai/dev/) for the version of
the documentation, which contains the unreleased features.

## Problem Domain

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

- Stiff and non-stiff universal ordinary differential equations (universal ODEs)
- Universal stochastic differential equations (universal SDEs)
- Universal delay differential equations (universal DDEs)
- Universal partial differential equations (universal PDEs)
- Universal jump stochastic differential equations (universal jump diffusions)
- Hybrid universal differential equations (universal DEs with event handling)

with high order, adaptive, implicit, GPU-accelerated, Newton-Krylov, etc.
methods. For examples, please refer to
[the release blog post](https://julialang.org/blog/2019/01/fluxdiffeq).
Additional demonstrations, like neural
PDEs and neural jump SDEs, can be found
[in this blog post](http://www.stochasticlifestyle.com/neural-jump-sdes-jump-diffusions-and-neural-pdes/)
(among many others!).

Do not limit yourself to the current neuralization. With this package, you can
explore various ways to integrate the two methodologies:

- Neural networks can be defined where the “activations” are nonlinear functions
  described by differential equations
- Neural networks can be defined where some layers are ODE solves
- ODEs can be defined where some terms are neural networks
- Cost functions on ODEs can define neural networks

![Flux ODE Training Animation](https://user-images.githubusercontent.com/1814174/88589293-e8207f80-d026-11ea-86e2-8a3feb8252ca.gif)
