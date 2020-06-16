abstract type AbstractTensorProductLayer <: Function end

struct TensorProductLayer{Array{TensorProductBasis},S<:AbstractArray,b<:AbstractArray,Int} <: TensorProductLayer
    model::Array{TensorProductBasis}
    W::S
    b::T
    in::Int
    out::Int
    function TPLayer(in, out,model,W=nothing,b=nothing)
        number_of_weights = 1
        for basis in model:
            number_of_weights = number_of_weights*basis.n
        end
        W = [[zeros(number_of_weights)] for k in 1:out]
        b = [[0f0] for k in 1:out]
        new{typeof(W),typeof(b),typeof(in),typeof(out)}(W,b,in,out)
    end
end

function (layer::TPLayer)(x)
    z = []
    tensor_prod = model[1](x)
    for i in 2:end
        tensor_prod = kron(tensor_prod, model[i](x))
    end
    for i in 1:out
        push!(z, sum(W[i].*tensor_prod)+b[i])
    end
    return z
end
