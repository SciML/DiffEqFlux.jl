abstract type TensorProductBasis <: Function end

##Chebyshev basis of the form [T_0(x), T_1(x), ..., T_n(x)] where T is the Chebyshev polynomial of the first kind
struct ChebyshevBasis{Int} <: TensorProductBasis
    n::Int
    function ChebyshevBasis(n)
        new{typeof(n)}(n)
    end
end

function (basis::ChebyshevBasis)(x)
    return [cos(j*acos(x)) for j in 1:basis.n]
end

##Fourier basis of the form F_j(x) = j is odd ? cos((j÷2)*x) : sin((j÷2)*x) => [F_0(x), F_1(x), ..., F_n(x)]
struct FourierBasis{Int} <: TensorProductBasis
    n::Int
    function FourierBasis(n)
        new{typeof(n)}(n)
    end
end

function (basis::FourierBasis)(x)
    return [sin(i*x) for i in 1:basis.n]
end

##Legendre basis of the form [P_0(x), P_1(x), ..., P_n(x)]
struct LegendreBasis{Int} <: TensorProductBasis
    n::Int
    function LegendreBasis(n)
        new{typeof(n)}(n)
    end
end

##Source: https://github.com/ranocha/PolynomialBases.jl/blob/master/src/legendre.jl
function legendre_poly(x, p::Integer)
    a::typeof(x) = one(x)
    b::typeof(x) = x

    if p <= 0
        return a
    elseif p == 1
        return b
    end

    for j in 2:p
        a, b = b, ( (2j-1)*x*b - (j-1)*a ) / j
    end

    b
end

function (basis::LegendreBasis)(x)
    f = k -> legendre_poly(x,k)
    return map(f, 1:basis.n)
end

##Polynomial basis of them form [x, x^2, ..., x^n]
struct PolynomialBasis{Int} <: TensorProductBasis
    n::Int

    function PolynomialBasis(n)
        new{typeof(n)}(n)
    end
end

function (basis::PolynomialBasis)(x)
    return [evalpoly(x, (I+zeros(basis.n,basis.n))[k,:]) for k in 1:basis.n]
end
