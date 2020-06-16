abstract type AbstractTensorProductLayer <: Function end
struct TensorProductLayer{M<:Array{TensorProductBasis},S<:AbstractArray,T<:AbstractArray,Int} <: AbstractTensorProductLayer
    model::M
    W::S
    b::T
    in::Int
    out::Int
    function TensorProductLayer(model,out,W=nothing,b=nothing)
        number_of_weights = 1
        for basis in model
            number_of_weights *= basis.n
        end
        W = zeros(out,number_of_weights)
        b = zeros(number_of_weights)
        new{typeof(model),typeof(W),typeof(b),Int}(model,W,b,length(model),out)
    end
end

function (layer::TensorProductLayer)(x)
    model, W, b = layer.model, layer.W, layer.b
    z = []
    tensor_prod = model[1](x)
    for i in 2:length(model)
        tensor_prod = kron(tensor_prod,model[i](x[i]))
    end
    for i in 1:layer.out
        push!(z, sum(W[i,:].*tensor_prod)+b[i])
    end
    return z
end
