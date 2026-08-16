# Neural Differential Equation Layer Functions

The following layers are helper functions for easily building neural differential
equation architectures in the currently most efficient way. As demonstrated in the
tutorials, they do not have to be used since automatic differentiation will
just work over `solve`, but these cover common use cases and choose
what's known to be the optimal mode of AD for the respective equation type.

```@docs
DiffEqFlux.NeuralDELayer
DiffEqFlux.NeuralSDELayer
NeuralODE
NeuralDSDE
NeuralSDE
NeuralCDDE
NeuralDAE
NeuralODEMM
AugmentedNDELayer
```

# Helper Layer Functions

```@docs
DimMover
```

## Adjoint APIs

DiffEqFlux explicitly reexports the following solver-facing sensitivity algorithms.
Their implementation and full contract are maintained by SciMLSensitivity.jl.

The reexported names are:

[`AdjointLSS`](https://docs.sciml.ai/SciMLSensitivity/stable/),
[`BacksolveAdjoint`](https://docs.sciml.ai/SciMLSensitivity/stable/),
[`EnzymeVJP`](https://docs.sciml.ai/SciMLSensitivity/stable/),
[`ForwardDiffOverAdjoint`](https://docs.sciml.ai/SciMLSensitivity/stable/),
[`ForwardDiffSensitivity`](https://docs.sciml.ai/SciMLSensitivity/stable/),
[`ForwardLSS`](https://docs.sciml.ai/SciMLSensitivity/stable/),
[`ForwardSensitivity`](https://docs.sciml.ai/SciMLSensitivity/stable/),
[`GaussAdjoint`](https://docs.sciml.ai/SciMLSensitivity/stable/),
[`InterpolatingAdjoint`](https://docs.sciml.ai/SciMLSensitivity/stable/),
[`NILSAS`](https://docs.sciml.ai/SciMLSensitivity/stable/),
[`NILSS`](https://docs.sciml.ai/SciMLSensitivity/stable/),
[`QuadratureAdjoint`](https://docs.sciml.ai/SciMLSensitivity/stable/),
[`ReverseDiffAdjoint`](https://docs.sciml.ai/SciMLSensitivity/stable/),
[`ReverseDiffVJP`](https://docs.sciml.ai/SciMLSensitivity/stable/),
[`SteadyStateAdjoint`](https://docs.sciml.ai/SciMLSensitivity/stable/),
[`TrackerAdjoint`](https://docs.sciml.ai/SciMLSensitivity/stable/),
[`TrackerVJP`](https://docs.sciml.ai/SciMLSensitivity/stable/),
[`ZygoteAdjoint`](https://docs.sciml.ai/SciMLSensitivity/stable/), and
[`ZygoteVJP`](https://docs.sciml.ai/SciMLSensitivity/stable/).
