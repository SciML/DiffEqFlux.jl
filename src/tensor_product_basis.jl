abstract type TensorProductBasis <: Function end

"""
Constructs a Chebyshev basis of the form [T_{0}(x), T_{1}(x), ..., T_{n-1}(x)] where T_j(.) is the j-th Chebyshev polynomial of the first kind.
```julia
ChebyshevBasis(n)
```
Arguments:
- `n`: number of terms in the polynomial expansion.
"""
struct ChebyshevBasis <: TensorProductBasis
    n::Int
end

function (basis::ChebyshevBasis)(x)
    return map(j -> cos(j*acos(x)), 1:basis.n)
end

"""
Constructs a sine basis of the form [sin(x), sin(2*x), ..., sin(n*x)].
```julia
SinBasis(n)
```
Arguments:
- `n`: number of terms in the sine expansion.
"""
struct SinBasis <: TensorProductBasis
    n::Int
end

function (basis::SinBasis)(x)
    return map(j -> sin(j*x), 1:basis.n)
end

"""
Constructs a cosine basis of the form [cos(x), cos(2*x), ..., cos(n*x)].
```julia
CosBasis(n)
```
Arguments:
- `n`: number of terms in the cosine expansion.
"""
struct CosBasis <: TensorProductBasis
    n::Int
end

function (basis::CosBasis)(x)
    return map(j -> cos(j*x), 1:basis.n)
end

#auxiliary function
function fourier(i::Int, x::Real)
    return iseven(i) ? cos(i*x/2) : sin(i*x/2)
end

"""
Constructs a Fourier basis of the form F_j(x) = j is even ? cos((j÷2)*x) : sin((j÷2)*x) => [F_0(x), F_1(x), ..., F_n(x)].
```julia
FourierBasis(n)
```
Arguments:
- `n`: number of terms in the Fourier expansion.
"""
struct FourierBasis <: TensorProductBasis
    n::Int
end

function (basis::FourierBasis)(x)
    return map(j -> fourier(j,x), 1:basis.n)
end

#auxiliary function
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

"""
Constructs a Legendre basis of the form [P_{0}(x), P_{1}(x), ..., P_{n-1}(x)] where P_j(.) is the j-th Legendre polynomial.
```julia
LegendreBasis(n)
```
Arguments:
- `n`: number of terms in the polynomial expansion.
"""
struct LegendreBasis <: TensorProductBasis
    n::Int
end

function (basis::LegendreBasis)(x)
    f = k -> legendre_poly(x,k-1)
    return map(f, 1:basis.n)
end

"""
Constructs a Polynomial basis of the form [1, x, ..., x^(n-1)].
```julia
PolynomialBasis(n)
```
Arguments:
- `n`: number of terms in the polynomial expansion.
"""
struct PolynomialBasis <: TensorProductBasis
    n::Int
end

function (basis::PolynomialBasis)(x)
    return [evalpoly(x, (I+zeros(basis.n,basis.n))[k,:]) for k in 1:basis.n]
end
