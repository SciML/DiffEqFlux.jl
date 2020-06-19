abstract type AbstractTensorProductLayer <: Function end
struct TensorLayer{M<:Array{TensorProductBasis},ComponentArray,Int} <: AbstractTensorProductLayer
    model::M
    p::ComponentArray
    in::Int
    out::Int
    function TensorLayer(model,out,W=nothing,b=nothing)
        number_of_weights = 1
        for basis in model
            number_of_weights *= basis.n
        end
        W = zeros(out, number_of_weights)
        b = zeros(out)
        p = ComponentArray(W = W, b = b)
        new{Array{TensorProductBasis},typeof(p),Int}(model,p,length(model),out)
    end
end

function (layer::TensorLayer)(x,p=layer.p)
    model,out = layer.model,layer.out
    W = @view p.W[:,:]
    b = @view p.b[:,:]
    tensor_prod = model[1](x[1])
    for i in 2:length(model)
        tensor_prod = kron(tensor_prod,model[i](x[i]))
    end
    z = [sum(W[i,:].*tensor_prod).+b[i] for i in 1:out]
    return Float32.(z)
end
