children(x) = ()
mapchildren(f, x) = x

children(x::Tuple) = x
children(x::NamedTuple) = x
mapchildren(f, x::Tuple) = map(f, x)
mapchildren(f, x::NamedTuple) = map(f, x)

function treelike(m::Module, T, fs = fieldnames(T))
  @eval m begin
    Flux.children(x::$T) = ($([:(x.$f) for f in fs]...),)
    Flux.mapchildren(f, x::$T) = $T(f.($children(x))...)
  end
end

macro treelike(T, fs = nothing)
  fs == nothing || isexpr(fs, :tuple) || error("@treelike T (a, b)")
  fs = fs == nothing ? [] : [:($(map(QuoteNode, fs.args)...),)]
  :(treelike(@__MODULE__, $(esc(T)), $(fs...)))
end

isleaf(x) = isempty(children(x))

function mapleaves(f, x; cache = IdDict())
  haskey(cache, x) && return cache[x]
  cache[x] = isleaf(x) ? f(x) : mapchildren(x -> mapleaves(f, x, cache = cache), x)
end

function destructure(m)
  xs = []
  mapleaves(m) do x
    x isa TrackedArray && push!(xs, x)
    return x
  end
  return vcat(vec.(xs)...)
end

function restructure(m, xs)
  i = 0
  mapleaves(m) do x
    x isa TrackedArray || return x
    x = reshape(xs[i.+(1:length(x))], size(x))
    i += length(x)
    return x
  end
end
