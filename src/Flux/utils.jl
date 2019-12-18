using Flux.Zygote

function params_(m)
  i = 0
  Flux.fmap(m) do x
    x isa AbstractArray && (i += 1)
    return x
  end
  xs = Zygote.Buffer(Vector(undef, i))
  i = 1
  Flux.fmap(m) do x
    if x isa AbstractArray
      xs[i] = x
      i += 1
    end
    x
  end
  copy(xs)
end

function destructure(m)
  vcat(vec.(params_(m))...)
end

function restructure(m, xs)
  i = 0
  Flux.mapleaves(m) do x
    x isa AbstractArray || return x
    x = reshape(xs[i.+(1:length(x))], size(x))
    i += length(x)
    return x
  end
end
