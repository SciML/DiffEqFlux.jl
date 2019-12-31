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

@generated function restructure(sz::Tuple, xs)
  gen = (:(reshape(xs[(sum(prod.(sz[(1:$(i-1))]))+1):sum(prod.(sz[1:$i]))],sz[$i])) for i in 2:length(sz.parameters))
  Expr(:vect,:(reshape(xs[1:sum(prod(sz[1]))], sz[1])),gen...)
end

#=
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
=#
