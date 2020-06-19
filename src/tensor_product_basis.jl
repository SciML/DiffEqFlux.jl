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

function (basis::LegendreBasis)(x)
    #TODO: fix Legendre Basis
    A = ones(basis.n)
    return A
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
