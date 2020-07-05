abstract type AbstractTensorProductLayer <: Function end
"""
Constructs the Tensor Product Layer, which takes as input an array of n tensor
product basis, [B_1, B_2, ..., B_n] a data point x, computes
z[i] = W[i,:] ⨀ [B_1(x[1]) ⨂ B_2(x[2]) ⨂ ... ⨂ B_n(x[n])], where W is the layer's weight,
and returns [z[1], ..., z[out]].

```julia
TensorLayer(model,out,p=nothing)
```
Arguments:
- `model`: Array of TensorProductBasis [B_1(n_1), ..., B_k(n_k)], where k corresponds to the dimension of the input.
- `out`: Dimension of the output.
- `p`: Optional initialization of the layer's weight. Initizalized to 0 by default.
"""
struct TensorLayer{M<:Array{TensorProductBasis},P<:AbstractArray,Int} <: AbstractTensorProductLayer
    model::M
    p::P
    in::Int
    out::Int
    function TensorLayer(model,out,p=nothing)
        number_of_weights = 1
        for basis in model
            number_of_weights *= basis.n
        end
        p = randn(out*number_of_weights)
        new{Array{TensorProductBasis},typeof(p),Int}(model,p,length(model),out)
    end
end

function (layer::TensorLayer)(x,p=layer.p)
    model,out = layer.model,layer.out
    W = reshape(p, Int(length(p)/out), out)
    tensor_prod = model[1](x[1])
    for i in 2:length(model)
        tensor_prod = kron(tensor_prod,model[i](x[i]))
    end
    z = W*tensor_prod
    return z
end
