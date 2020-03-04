function __init__()
    @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
        gpu_or_cpu(x::CuArrays.CuArray) = CuArrays.CuArray
        gpu_or_cpu(x::TrackedArray{<:Any,<:Any,<:CuArrays.CuArray}) = CuArrays.CuArray
        gpu_or_cpu(x::Transpose{<:Any,<:CuArrays.CuArray}) = CuArrays.CuArray
        gpu_or_cpu(x::Adjoint{<:Any,<:CuArrays.CuArray}) = CuArrays.CuArray
        gpu_or_cpu(x::Adjoint{<:Any,TrackedArray{<:Any,<:Any,<:CuArrays.CuArray}}) = CuArrays.CuArray
        gpu_or_cpu(x::Transpose{<:Any,TrackedArray{<:Any,<:Any,<:CuArrays.CuArray}}) = CuArrays.CuArray
    end
    
    @require NLopt="76087f3c-5699-56af-9a33-bf431cd00edd" begin
        function sciml_train(loss, θ, opt::NLopt.Opt, data = DEFAULT_DATA; maxeval=100)
            local x, cur, state
            cur,state = iterate(data)
        
            function nlopt_grad!(θ,grad)
              _x,lambda = Flux.Zygote.pullback(θ) do θ
                x = loss(θ)
                first(x)
              end
          
              if length(grad) > 0 
                grad .= first(lambda(1))
              end
          
              return _x
            end
        
            NLopt.min_objective!(opt, nlopt_grad!)
            NLopt.maxeval!(opt, maxeval)
            NLopt.optimize(opt, θ)
        end
      
        function sciml_train(loss, θ, opt::NLopt.Opt, lower_bounds, upper_bounds, data = DEFAULT_DATA; maxeval=100)
          local x, cur, state
          cur,state = iterate(data)
        
          function nlopt_grad!(θ,grad)
            _x,lambda = Flux.Zygote.pullback(θ) do θ
              x = loss(θ)
              first(x)
            end
        
            if length(grad) > 0 
              grad .= first(lambda(1))
            end
        
            return _x
          end
      
          NLopt.min_objective!(opt, nlopt_grad!)
          NLopt.lower_bounds!(opt, lower_bounds)
          NLopt.upper_bounds!(opt, upper_bounds)
          NLopt.maxeval!(opt, maxeval)
          NLopt.optimize(opt, θ)
        end
    end
end