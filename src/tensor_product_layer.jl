abstract type AbstractTensorProductLayer <: Function end
struct TensorProductLayer{M<:Array{TensorProductBasis},ComponentArray,Int} <: AbstractTensorProductLayer
    model::M
    component::ComponentArray
    in::Int
    out::Int
    function TensorProductLayer(model,out,W=nothing,b=nothing)
        number_of_weights = 1
        for basis in model
            number_of_weights *= basis.n
        end
        w = zeros(out, number_of_weights)
        b = zeros(out)
        component = ComponentArray(W = w, B = b)
        new{typeof(model),typeof(component),Int}(model,component,length(model),out)
    end
end

function (layer::TensorProductLayer)(x,component=layer.component)
    model,out = layer.model,layer.out
    w = @view component.W[:,:]
    b = @view component.B[:,:]
    tensor_prod = model[1](x[1])
    for i in 2:length(model)
        tensor_prod = kron(tensor_prod,model[i](x[i]))
    end
    z = [sum(w[i,:].*tensor_prod).+b[i] for i in 1:out]
    return Float32.(z)
end
