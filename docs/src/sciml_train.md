# [sciml_train and GalacticOptim.jl](@id sciml_train)

`sciml_train` is a heuristic-based training function built using GalacticOptim.jl.
It incorporates the knowledge of many high level benchmarks to attempt and do
the right thing.

GalacticOptim.jl is a package with a scope that is beyond your normal global optimization
package. GalacticOptim.jl seeks to bring together all of the optimization packages
it can find, local and global, into one unified Julia interface. This means, you
learn one package and you learn them all! GalacticOptim.jl adds a few high-level
features, such as integrating with automatic differentiation, to make its usage
fairly simple for most cases, while allowing all of the options in a single
unified interface.

## sciml_train API

```@docs
DiffEqFlux.sciml_train
```
