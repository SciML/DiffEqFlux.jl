abstract type TensorProductBasis <: Function end

##TODO: Chebyshev Basis

##Fourier basis of the form F_j(x) = j is odd ? cos((j÷2)*x) : sin((j÷2)*x) => [F_0(x), F_1(x), ..., F_n(x)]
struct FourierBasis{Int} <: TensorProductBasis
    n::Int

    function FourierBasis(n)
        new{typeof(n)}(n)
    end
end

function (basis::FourierBasis)(x)
    return Fourier()[x, 1:basis.n]
end

##Legendre basis of the form [P_0(x), P_1(x), ..., P_n(x)]
struct LegendreBasis{Int} <: TensorProductBasis
    n::Int

    function LegendreBasis(n)
        new{typeof(n)}(n)
    end
end

function (basis::LegendreBasis)(x)
    return Legendre()[x, 1:basis.n]
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
