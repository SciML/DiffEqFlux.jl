# Reexported API

`using DiffEqFlux` brings a small, deliberate slice of its dependencies into scope on
top of DiffEqFlux's own layers, so that the examples in this documentation can build a
neural differential equation end to end without a second `using`. Every name below is
**owned and documented by the upstream package** -- DiffEqFlux only re-exports it, and
the upstream documentation is where to look for what each one does.

Anything not on this page must be imported from its own package. In particular, the
canonical way to work with DiffEqFlux is still

```julia
using DiffEqFlux, Lux
```

which gives the full Lux surface; the re-exports here only cover what the DiffEqFlux
documentation itself uses.

## Network construction ([Lux.jl](https://lux.csail.mit.edu/stable/))

The `Lux` module itself is re-exported (so `Lux.setup`, `Lux.Chain`, ... work), along
with the layers used by this documentation:

  - Containers and basic layers: `Chain`, `Dense`, `WrappedFunction`
  - Convolutional and pooling layers: `Conv`, `MaxPool`, `MeanPool`, `FlattenLayer`,
    `SamePad`
  - Normalization: `GroupNorm`
  - Stateful wrapper: `StatefulLuxLayer`
  - Flux interop: `FromFluxAdaptor`
  - Precision helpers: `f32`, `f64`
  - The `Training` module

The rest of Lux -- its full layer zoo, loss functions, activation functions (owned by
NNlib), weight initializers (owned by WeightInitializers) and device helpers (owned by
MLDataDevices) -- is **not** re-exported. Use `using Lux` for those.

## Layer contract ([LuxCore.jl](https://lux.csail.mit.edu/stable/api/Building_Blocks/LuxCore))

The abstract types you subtype to write your own layer, and the functions that layer
must support:

  - `AbstractLuxLayer`, `AbstractLuxContainerLayer`, `AbstractLuxWrapperLayer`
  - `setup`, `apply`, `initialparameters`, `initialstates`, `parameterlength`,
    `statelength`, `testmode`, `trainmode`
  - the `LuxCore` module itself

## Differentiation backends ([ADTypes.jl](https://sciml.github.io/ADTypes.jl/stable/))

The backend selectors passed to `ad =` on layers such as [`FFJORD`](@ref) and to
`Optimization.OptimizationFunction`:

`AutoChainRules`, `AutoDiffractor`, `AutoEnzyme`, `AutoFastDifferentiation`,
`AutoFiniteDiff`, `AutoFiniteDifferences`, `AutoForwardDiff`, `AutoGTPSA`,
`AutoHyperHessians`, `AutoModelingToolkit`, `AutoMooncake`, `AutoMooncakeForward`,
`AutoPolyesterForwardDiff`, `AutoReactant`, `AutoReverseDiff`, `AutoSparse`,
`AutoSymbolics`, `AutoTaylorDiff`, `AutoTracker`, `AutoZygote`, plus `AbstractADType`
and the `ADTypes` module.

ADTypes' sparsity-detection and coloring interfaces are not re-exported; use
`using ADTypes` for those.

## Model zoo ([Boltz.jl](https://luxdl.github.io/Boltz.jl/stable/))

  - `Layers` -- e.g. `Layers.HamiltonianNN`, `Layers.TensorProductLayer`
  - `Basis` -- e.g. `Basis.Legendre`
  - the `Boltz` module itself

Boltz's `Vision` and `PIML` submodules are not re-exported; use `using Boltz` for
those.

## Sensitivity analysis ([SciMLSensitivity.jl](https://docs.sciml.ai/SciMLSensitivity/stable/))

The adjoint and vector-Jacobian-product choices passed through to the solver:

  - Adjoints: `BacksolveAdjoint`, `QuadratureAdjoint`, `GaussAdjoint`,
    `InterpolatingAdjoint`, `TrackerAdjoint`, `ZygoteAdjoint`, `ReverseDiffAdjoint`,
    `SteadyStateAdjoint`, `ForwardDiffOverAdjoint`
  - Forward sensitivity: `ForwardSensitivity`, `ForwardDiffSensitivity`
  - Shadowing methods: `ForwardLSS`, `AdjointLSS`, `NILSS`, `NILSAS`
  - VJP choices: `TrackerVJP`, `ZygoteVJP`, `EnzymeVJP`, `ReverseDiffVJP`

Anything else from SciMLSensitivity must be imported from SciMLSensitivity directly.
