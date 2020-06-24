# Tensor Product Layer

The following layer is a helper function for easily constructing a TensorLayer, which
take as input an array of n tensor product basis, [B_1, B_2, ..., B_n], a data
point x, computes z[i] = W[i,:] ⨀ [B_1(x[1]) ⨂ B_2(x[2]) ⨂ ... ⨂ B_n(x[n])],
where W is the layer's weight, and returns [z[1], ..., z[out]].

```@docs
TensorLayer
```
