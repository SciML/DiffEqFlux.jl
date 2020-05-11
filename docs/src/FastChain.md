# FastChain

The `FastChain` system is a Flux-like explicit parameter neural network
architecture system for less overhead in smaller neural networks. For neural
networks with layers of lengths >~200, these optimizations are overshadowed by
the cost of matrix multiplication. However, for smaller layer operations this
architecture can reduce a lot of the overhead traditionally seen in neural
network architectures and thus is recommended in a lot of scientific machine
learning usecaes.

## Basics

The basic is that `FastChain` is a collection of functions of two values,
`(x,p)`, and chains these functions to call one after the next. Each layer in
this chain gets pre-defined amount of parameters sent to it. For example,

```julia
f = FastChain((x,p) -> x.^3,
              FastDense(2,50,tanh),
              FastDense(50,2))
```

`FastChain` here has a `2*50 + 50` length parameter `FastDense(2,50,tanh)` function
and a `50*2 + 2` parameter function `FastDense(50,2)`. The first function gets
the default number of parameters which is 0. Thus `f(x,p)` is equivalent to the
following code:

```julia
function f(x,p)
  tmp1 = x.^3
  len1 = paramlength(FastDense(2,50,tanh))
  tmp2 = FastDense(2,50,tanh)(tmp1,@view p[1:len1])
  tmp3 = FastDense(50,2)(tmp2,@view p[len2:end])
end
```

`FastChain` functions thus require that the vector of neural network parameters
is passed to it on each call, making the setup explicit in the passed parameters.

To get initial parameters for the optimization of a function defined by a
`FastChain`, one simply calls `initial_params(f)` which returns the concatenation
of the initial parameters for each layer. Notice that since all parameters are
explicit, constructing and reconstructing chains/layers can be a memory-free
operation, since the only memory is the parameter vector itself which is handled
by the user.

### FastChain Interface

The only requirement to be a layer in `FastChain` is to be a 2-argument function
`l(x,p)` and define the following traits:

- `paramlength(::typeof(l))`: The number of parameters from the parameter vector
  to allocate to this layer. Defaults to zero.
- `initial_params(::typeof(l))`: The function for defining the initial parameters
  of the layer. Should output a vector of length matching `paramlength`. Defaults
  to `Float32[]`.

## FastChain-Compatible Layers

The following pre-defined layers can be used with `FastChain`:

```@docs
FastDense
StaticDense
```
