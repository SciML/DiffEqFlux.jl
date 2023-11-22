# Neural Differential Equation Layer Functions

The following layers are helper functions for easily building neural differential
equation architectures in the currently most efficient way. As demonstrated in the
tutorials, they do not have to be used since automatic differentiation will
just work over `solve`, but these cover common use cases and choose
what's known to be the optimal mode of AD for the respective equation type.

```@docs
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
