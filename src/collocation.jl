abstract type CollocationKernel end
struct EpanechnikovKernel <: CollocationKernel end
struct UniformKernel <: CollocationKernel end
struct TriangularKernel <: CollocationKernel end
struct QuarticKernel <: CollocationKernel end
struct TriweightKernel <: CollocationKernel end
struct TricubeKernel <: CollocationKernel end
struct GaussianKernel <: CollocationKernel end
struct CosineKernel <: CollocationKernel end
struct LogisticKernel <: CollocationKernel end
struct SigmoidKernel <: CollocationKernel end
struct SilvermanKernel <: CollocationKernel end

function calckernel(::EpanechnikovKernel,t)
    if abs(t) > 1
        return 0
    else
        return 0.75*(1-t^2)
    end
end

function calckernel(::UniformKernel,t)
    if abs(t) > 1
        return 0
    else
        return 0.5
    end
end

function calckernel(::TriangularKernel,t)
    if abs(t) > 1
        return 0
    else
        return (1-abs(t))
    end
end

function calckernel(::QuarticKernel,t)
  if abs(t)>0
    return 0
  else
    return (15*(1-t^2)^2)/16
  end
end

function calckernel(::TriweightKernel,t)
  if abs(t)>0
    return 0
  else
    return (35*(1-t^2)^3)/32
  end
end

function calckernel(::TricubeKernel,t)
  if abs(t)>0
    return 0
  else
    return (70*(1-abs(t)^3)^3)/80
  end
end

function calckernel(::GaussianKernel,t)
  exp(-0.5*t^2)/(sqrt(2*π))
end

function calckernel(::CosineKernel,t)
  if abs(t)>0
    return 0
  else
    return (π*cos(π*t/2))/4
  end
end

function calckernel(::LogisticKernel,t)
  1/(exp(t)+2+exp(-t))
end

function calckernel(::SigmoidKernel,t)
  2/(π*(exp(t)+exp(-t)))
end

function calckernel(::SilvermanKernel,t)
  sin(abs(t)/2+π/4)*0.5*exp(-abs(t)/sqrt(2))
end

function construct_t1(t,tpoints)
    hcat(ones(eltype(tpoints),length(tpoints)),tpoints.-t)
end

function construct_t2(t,tpoints)
  hcat(ones(eltype(tpoints),length(tpoints)),tpoints.-t,(tpoints.-t).^2)
end

function construct_w(t,tpoints,h,kernel)
    W = @. calckernel((kernel,),(tpoints-t)/h)/h
    Matrix(Diagonal(W))
end


"""
```julia
u′,u = collocate_data(data,tpoints,kernel=SigmoidKernel())
```

Computes a non-parametrically smoothed estimate of `u'` and `u`
given the `data`, where each column is a snapshot of the timeseries at
`tpoints[i]`.

For kernels, the following exist:

- EpanechnikovKernel
- UniformKernel
- TriangularKernel
- QuarticKernel
- TriweightKernel
- TricubeKernel
- GaussianKernel
- CosineKernel
- LogisticKernel
- SigmoidKernel
- SilvermanKernel

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2631937/
"""
function collocate_data(data,tpoints,kernel=TriangularKernel())
  _one = oneunit(first(data))
  _zero = zero(first(data))
  e1 = [_one;_zero]
  e2 = [_zero;_one;_zero]
  n = length(tpoints)
  h = (n^(-1/5))*(n^(-3/35))*((log(n))^(-1/16))

  x = map(tpoints) do _t
      T1 = construct_t1(_t,tpoints)
      T2 = construct_t2(_t,tpoints)
      W = construct_w(_t,tpoints,h,kernel)
      e2'*inv(T2'*W*T2)T2'*W*data',e1'*inv(T1'*W*T1)*T1'*W*data'
  end
  estimated_derivative = reduce(hcat,transpose.(first.(x)))
  estimated_solution = reduce(hcat,transpose.(last.(x)))
  estimated_derivative,estimated_solution
end
