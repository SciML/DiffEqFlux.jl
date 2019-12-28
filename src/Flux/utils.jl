function destructure(m)
  xs = []
  Flux.mapleaves(m) do x
    x isa AbstractArray && push!(xs, x)
    return x
  end
  return vcat(vec.(xs)...)
end

#=
ZygoteRules.@adjoint function destructure(m)
  xs = []
  Flux.mapleaves(m) do x
    x isa AbstractArray && push!(xs, x)
    return x
  end
  vcat(vec.(xs)...),ybar->(nothing,)
end
=#

function restructure(m, xs)
  i = 0
  Flux.mapleaves(m) do x
    x isa AbstractArray || return x
    x = reshape(xs[i.+(1:length(x))], size(x))
    i += length(x)
    return x
  end
end

function restructure(sz::Tuple, xs)
  t = []
  i = 0
  for s in sz
    y = reshape(xs[i.+(1:prod(s))], s)
    push!(t,y)
    i += length(y)
  end
  t
end
