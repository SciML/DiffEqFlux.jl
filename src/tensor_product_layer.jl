abstract type AbstractTensorProductLayer <: Function end
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
        p = rand(out*number_of_weights)
        new{Array{TensorProductBasis},typeof(p),Int}(model,p,length(model),out)
    end
end

function (layer::TensorLayer)(x,p=layer.p)
    model,out = layer.model,layer.out
    W = reshape(p, Int(length(p)/out), out)'
    tensor_prod = model[1](x[1])
    for i in 2:length(model)
        tensor_prod = kron(tensor_prod,model[i](x[i]))
    end
    z = [sum(W[i,:].*tensor_prod) for i in 1:out]
    return z
end
