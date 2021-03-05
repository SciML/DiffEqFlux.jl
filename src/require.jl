isgpu(x) = false
ifgpufree(x) = nothing
function __init__()
    @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
        gpu_or_cpu(x::CuArrays.CuArray) = CuArrays.CuArray
        gpu_or_cpu(x::Transpose{<:Any,<:CuArrays.CuArray}) = CuArrays.CuArray
        gpu_or_cpu(x::Adjoint{<:Any,<:CuArrays.CuArray}) = CuArrays.CuArray
        isgpu(::CuArrays.CuArray) = true
        isgpu(::Transpose{<:Any,<:CuArrays.CuArray}) = true
        isgpu(::Adjoint{<:Any,<:CuArrays.CuArray}) = true
        ifgpufree(x::CuArrays.CuArray) = CuArrays.unsafe_free!(x)
        ifgpufree(x::Transpose{<:Any,<:CuArrays.CuArray}) = CuArrays.unsafe_free!(x.parent)
        ifgpufree(x::Adjoint{<:Any,<:CuArrays.CuArray}) = CuArrays.unsafe_free!(x.parent)

        @require Tracker="9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" begin
            TrackedArray = Tracker.TrackedArray
            gpu_or_cpu(x::TrackedArray{<:Any,<:Any,<:CuArrays.CuArray}) = CuArrays.CuArray
            gpu_or_cpu(x::Adjoint{<:Any,TrackedArray{<:Any,<:Any,<:CuArrays.CuArray}}) = CuArrays.CuArray
            gpu_or_cpu(x::Transpose{<:Any,TrackedArray{<:Any,<:Any,<:CuArrays.CuArray}}) = CuArrays.CuArray
            isgpu(::TrackedArray{<:Any,<:Any,<:CuArrays.CuArray}) = true
            ifgpufree(x::TrackedArray{<:Any,<:Any,<:CuArrays.CuArray}) = CuArrays.unsafe_free!(x.data)
            isgpu(::Adjoint{<:Any,TrackedArray{<:Any,<:Any,<:CuArrays.CuArray}}) = true
            isgpu(::Transpose{<:Any,TrackedArray{<:Any,<:Any,<:CuArrays.CuArray}}) = true
            ifgpufree(x::Adjoint{<:Any,TrackedArray{<:Any,<:Any,<:CuArrays.CuArray}}) = CuArrays.unsafe_free!((x.data).parent)
            ifgpufree(x::Transpose{<:Any,TrackedArray{<:Any,<:Any,<:CuArrays.CuArray}}) = CuArrays.unsafe_free!((x.data).parent)
        end
    end

    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
        gpu_or_cpu(x::CUDA.CuArray) = CUDA.CuArray
        gpu_or_cpu(x::Transpose{<:Any,<:CUDA.CuArray}) = CUDA.CuArray
        gpu_or_cpu(x::Adjoint{<:Any,<:CUDA.CuArray}) = CUDA.CuArray
        isgpu(::CUDA.CuArray) = true
        isgpu(::Transpose{<:Any,<:CUDA.CuArray}) = true
        isgpu(::Adjoint{<:Any,<:CUDA.CuArray}) = true
        ifgpufree(x::CUDA.CuArray) = CUDA.unsafe_free!(x)
        ifgpufree(x::Transpose{<:Any,<:CUDA.CuArray}) = CUDA.unsafe_free!(x.parent)
        ifgpufree(x::Adjoint{<:Any,<:CUDA.CuArray}) = CUDA.unsafe_free!(x.parent)

        @require Tracker="9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" begin
            TrackedArray = Tracker.TrackedArray
            gpu_or_cpu(x::TrackedArray{<:Any,<:Any,<:CUDA.CuArray}) = CUDA.CuArray
            gpu_or_cpu(x::Adjoint{<:Any,TrackedArray{<:Any,<:Any,<:CUDA.CuArray}}) = CUDA.CuArray
            gpu_or_cpu(x::Transpose{<:Any,TrackedArray{<:Any,<:Any,<:CUDA.CuArray}}) = CUDA.CuArray
            isgpu(::Adjoint{<:Any,TrackedArray{<:Any,<:Any,<:CUDA.CuArray}}) = true
            isgpu(::TrackedArray{<:Any,<:Any,<:CUDA.CuArray}) = true
            isgpu(::Transpose{<:Any,TrackedArray{<:Any,<:Any,<:CUDA.CuArray}}) = true
            ifgpufree(x::TrackedArray{<:Any,<:Any,<:CUDA.CuArray}) = CUDA.unsafe_free!(x.data)
            ifgpufree(x::Adjoint{<:Any,TrackedArray{<:Any,<:Any,<:CUDA.CuArray}}) = CUDA.unsafe_free!((x.data).parent)
            ifgpufree(x::Transpose{<:Any,TrackedArray{<:Any,<:Any,<:CUDA.CuArray}}) = CUDA.unsafe_free!((x.data).parent)
        end
    end
end
