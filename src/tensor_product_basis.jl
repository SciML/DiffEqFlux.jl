abstract type TensorProductBasis <: Function end

##Chebyshev basis of the form [T_0(x), T_1(x), ..., T_n(x)] where T is the Chebyshev polynomial of the first kind
struct ChebyshevBasis <: TensorProductBasis
    n::Int
end

function (basis::ChebyshevBasis)(x)
    return [cos(j*acos(x)) for j in 1:basis.n]
end

##Sine basis of the form [sin(x), sin(2*x), ..., sin(n*x)]
struct SinBasis <: TensorProductBasis
    n::Int
end

function (basis::SinBasis)(x)
    return [sin(i*x) for i in 1:basis.n]
end

##Cosine basis of the form [cos(x), cos(2*x), ..., cos(n*x)]
struct CosBasis <: TensorProductBasis
    n::Int
end

function (basis::CosBasis)(x)
    return [cos(i*x) for i in 1:basis.n]
end

##Fourier basis of the form F_j(x) = j is even ? cos((j÷2)*x) : sin((j÷2)*x) => [F_0(x), F_1(x), ..., F_n(x)]
struct FourierBasis <: TensorProductBasis
    n::Int
end

function fourier(i, x)
    return iseven(i) ? cos(i*x/2) : sin(i*x/2)
end

function (basis::FourierBasis)(x)
    return [fourier(i, x) for i in 1:basis.n]
end

##Legendre basis of the form [P_0(x), P_1(x), ..., P_n(x)]
struct LegendreBasis <: TensorProductBasis
    n::Int
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
struct PolynomialBasis <: TensorProductBasis
    n::Int
end

function (basis::PolynomialBasis)(x)
    return [evalpoly(x, (I+zeros(basis.n,basis.n))[k,:]) for k in 1:basis.n]
end
