isgpu(x) = false
ifgpufree(x) = nothing
function __init__()
    @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
        gpu_or_cpu(x::CuArrays.CuArray) = CuArrays.CuArray
        gpu_or_cpu(x::TrackedArray{<:Any,<:Any,<:CuArrays.CuArray}) = CuArrays.CuArray
        gpu_or_cpu(x::Transpose{<:Any,<:CuArrays.CuArray}) = CuArrays.CuArray
        gpu_or_cpu(x::Adjoint{<:Any,<:CuArrays.CuArray}) = CuArrays.CuArray
        gpu_or_cpu(x::Adjoint{<:Any,TrackedArray{<:Any,<:Any,<:CuArrays.CuArray}}) = CuArrays.CuArray
        gpu_or_cpu(x::Transpose{<:Any,TrackedArray{<:Any,<:Any,<:CuArrays.CuArray}}) = CuArrays.CuArray
        isgpu(::CuArrays.CuArray) = true
        isgpu(::TrackedArray{<:Any,<:Any,<:CuArrays.CuArray}) = true
        isgpu(::Transpose{<:Any,<:CuArrays.CuArray}) = true
        isgpu(::Adjoint{<:Any,<:CuArrays.CuArray}) = true
        isgpu(::Adjoint{<:Any,TrackedArray{<:Any,<:Any,<:CuArrays.CuArray}}) = true
        isgpu(::Transpose{<:Any,TrackedArray{<:Any,<:Any,<:CuArrays.CuArray}}) = true
        ifgpufree(x::CuArrays.CuArray) = CuArrays.unsafe_free!(x)
        ifgpufree(x::TrackedArray{<:Any,<:Any,<:CuArrays.CuArray}) = CuArrays.unsafe_free!(x.data)
        ifgpufree(x::Transpose{<:Any,<:CuArrays.CuArray}) = CuArrays.unsafe_free!(x.parent)
        ifgpufree(x::Adjoint{<:Any,<:CuArrays.CuArray}) = CuArrays.unsafe_free!(x.parent)
        ifgpufree(x::Adjoint{<:Any,TrackedArray{<:Any,<:Any,<:CuArrays.CuArray}}) = CuArrays.unsafe_free!((x.data).parent)
        ifgpufree(x::Transpose{<:Any,TrackedArray{<:Any,<:Any,<:CuArrays.CuArray}}) = CuArrays.unsafe_free!((x.data).parent)
    end

    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
        gpu_or_cpu(x::CUDA.CuArray) = CUDA.CuArray
        gpu_or_cpu(x::TrackedArray{<:Any,<:Any,<:CUDA.CuArray}) = CUDA.CuArray
        gpu_or_cpu(x::Transpose{<:Any,<:CUDA.CuArray}) = CUDA.CuArray
        gpu_or_cpu(x::Adjoint{<:Any,<:CUDA.CuArray}) = CUDA.CuArray
        gpu_or_cpu(x::Adjoint{<:Any,TrackedArray{<:Any,<:Any,<:CUDA.CuArray}}) = CUDA.CuArray
        gpu_or_cpu(x::Transpose{<:Any,TrackedArray{<:Any,<:Any,<:CUDA.CuArray}}) = CUDA.CuArray
        isgpu(::CUDA.CuArray) = true
        isgpu(::TrackedArray{<:Any,<:Any,<:CUDA.CuArray}) = true
        isgpu(::Transpose{<:Any,<:CUDA.CuArray}) = true
        isgpu(::Adjoint{<:Any,<:CUDA.CuArray}) = true
        isgpu(::Adjoint{<:Any,TrackedArray{<:Any,<:Any,<:CUDA.CuArray}}) = true
        isgpu(::Transpose{<:Any,TrackedArray{<:Any,<:Any,<:CUDA.CuArray}}) = true
        ifgpufree(x::CUDA.CuArray) = CUDA.unsafe_free!(x)
        ifgpufree(x::TrackedArray{<:Any,<:Any,<:CUDA.CuArray}) = CUDA.unsafe_free!(x.data)
        ifgpufree(x::Transpose{<:Any,<:CUDA.CuArray}) = CUDA.unsafe_free!(x.parent)
        ifgpufree(x::Adjoint{<:Any,<:CUDA.CuArray}) = CUDA.unsafe_free!(x.parent)
        ifgpufree(x::Adjoint{<:Any,TrackedArray{<:Any,<:Any,<:CUDA.CuArray}}) = CUDA.unsafe_free!((x.data).parent)
        ifgpufree(x::Transpose{<:Any,TrackedArray{<:Any,<:Any,<:CUDA.CuArray}}) = CUDA.unsafe_free!((x.data).parent)
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
                @printf("fval:%2.2e  norm:%2.2e\n", _x[1], norm(θ))
              end

              return _x
            end

            NLopt.min_objective!(opt, nlopt_grad!)
            NLopt.maxeval!(opt, maxeval)

            t0= time()
            (minf,minx,ret) = NLopt.optimize(opt, θ)
            _time = time()

            Optim.MultivariateOptimizationResults(opt,
                                                    θ,# initial_x,
                                                    minx, #pick_best_x(f_incr_pick, state),
                                                    minf, # pick_best_f(f_incr_pick, state, d),
                                                    maxeval, #iteration,
                                                    maxeval >= maxeval, #iteration == options.iterations,
                                                    true, # x_converged,
                                                    0.0,#T(options.x_tol),
                                                    0.0,#T(options.x_tol),
                                                    NaN,# x_abschange(state),
                                                    NaN,# x_abschange(state),
                                                    true,# f_converged,
                                                    0.0,#T(options.f_tol),
                                                    0.0,#T(options.f_tol),
                                                    NaN,#f_abschange(d, state),
                                                    NaN,#f_abschange(d, state),
                                                    true,#g_converged,
                                                    0.0,#T(options.g_tol),
                                                    NaN,#g_residual(d),
                                                    false, #f_increased,
                                                    nothing,
                                                    maxeval,
                                                    maxeval,
                                                    0,
                                                    ret,
                                                    NaN,
                                                    _time-t0,)
        end

        function sciml_train(loss, θ, opt::NLopt.Opt, lower_bounds, upper_bounds, data = DEFAULT_DATA; maxeval=100, nstart=1)
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

          if nstart > 1
            localopt = opt
            opt = NLopt.Opt(:G_MLSL_LDS, length(lower_bounds))
            NLopt.lower_bounds!(opt, lower_bounds)
            NLopt.upper_bounds!(opt, upper_bounds)
            NLopt.local_optimizer!(opt, localopt)
            NLopt.maxeval!(opt, nstart * maxeval)
          end

          t0 = time()
          NLopt.optimize(opt, θ)
          _time = time()

          Optim.MultivariateOptimizationResults(opt,
                                                    θ,# initial_x,
                                                    minx, #pick_best_x(f_incr_pick, state),
                                                    minf, # pick_best_f(f_incr_pick, state, d),
                                                    maxeval, #iteration,
                                                    maxeval >= maxeval, #iteration == options.iterations,
                                                    true, # x_converged,
                                                    0.0,#T(options.x_tol),
                                                    0.0,#T(options.x_tol),
                                                    NaN,# x_abschange(state),
                                                    NaN,# x_abschange(state),
                                                    true,# f_converged,
                                                    0.0,#T(options.f_tol),
                                                    0.0,#T(options.f_tol),
                                                    NaN,#f_abschange(d, state),
                                                    NaN,#f_abschange(d, state),
                                                    true,#g_converged,
                                                    0.0,#T(options.g_tol),
                                                    NaN,#g_residual(d),
                                                    false, #f_increased,
                                                    nothing,
                                                    maxeval,
                                                    maxeval,
                                                    0,
                                                    ret,
                                                    NaN,
                                                    _time-t0,)
        end
    end

    @require MultistartOptimization = "3933049c-43be-478e-a8bb-6e0f7fd53575" begin
        function sciml_train(loss, _θ, opt::MultistartOptimization.TikTak, data = DEFAULT_DATA;lower_bounds, upper_bounds, local_method,
                              maxiters = get_maxiters(data), kwargs...)
          local x, cur, state
          cur,state = iterate(data)

          _loss = function (θ)
            x = loss(θ,cur...)
            first(x)
          end

          t0 = time()

          P = MultistartOptimization.MinimizationProblem(_loss, lower_bounds, upper_bounds)
          multistart_method = opt
          local_method = MultistartOptimization.NLoptLocalMethod(local_method)
          p = MultistartOptimization.multistart_minimization(multistart_method, local_method, P)

          t1 = time()

          Optim.MultivariateOptimizationResults(opt,
                                                [NaN],# initial_x,
                                                p.location, #pick_best_x(f_incr_pick, state),
                                                p.value, # pick_best_f(f_incr_pick, state, d),
                                                0, #iteration,
                                                false, #iteration == options.iterations,
                                                true, # x_converged,
                                                0.0,#T(options.x_tol),
                                                0.0,#T(options.x_tol),
                                                NaN,# x_abschange(state),
                                                NaN,# x_abschange(state),
                                                true,# f_converged,
                                                0.0,#T(options.f_tol),
                                                0.0,#T(options.f_tol),
                                                NaN,#f_abschange(d, state),
                                                NaN,#f_abschange(d, state),
                                                true,#g_converged,
                                                0.0,#T(options.g_tol),
                                                NaN,#g_residual(d),
                                                true, #f_increased,
                                                nothing,
                                                maxiters,
                                                maxiters,
                                                0,
                                                true,
                                                NaN,
                                                t1 - t0)
        end
    end
    @require QuadDIRECT = "dae52e8d-d666-5120-a592-9e15c33b8d7a" begin
        export QuadDirect
        struct QuadDirect
        end
        function sciml_train(loss, _θ, opt::QuadDirect, data = DEFAULT_DATA;lower_bounds, upper_bounds, splits,
                              maxiters = get_maxiters(data), kwargs...)
          local x, cur, state
          cur,state = iterate(data)

          _loss = function (θ)
            x = loss(θ,cur...)
            first(x)
          end

          t0 = time()

          root, x0 = QuadDIRECT.analyze(_loss, splits, lower_bounds, upper_bounds; kwargs...)

          t1 = time()

          Optim.MultivariateOptimizationResults(opt,
                                                [NaN],# initial_x,
                                                QuadDIRECT.position(root), #pick_best_x(f_incr_pick, state),
                                                QuadDIRECT.minimum(root), # pick_best_f(f_incr_pick, state, d),
                                                0, #iteration,
                                                false, #iteration == options.iterations,
                                                true, # x_converged,
                                                0.0,#T(options.x_tol),
                                                0.0,#T(options.x_tol),
                                                NaN,# x_abschange(state),
                                                NaN,# x_abschange(state),
                                                true,# f_converged,
                                                0.0,#T(options.f_tol),
                                                0.0,#T(options.f_tol),
                                                NaN,#f_abschange(d, state),
                                                NaN,#f_abschange(d, state),
                                                true,#g_converged,
                                                0.0,#T(options.g_tol),
                                                NaN,#g_residual(d),
                                                true, #f_increased,
                                                nothing,
                                                maxiters,
                                                maxiters,
                                                0,
                                                true,
                                                NaN,
                                                t1 - t0)
        end
    end
end
