abstract type TensorProductBasis <: Function end

function create_chebyshev_poly(n)
    poly_matrix = zeros(n, n)
    poly_matrix[1, 1] = 1
    poly_matrix[2, 2] = 1
    for k in 3:n
        poly_matrix[k,:] = 2.0.*circshift(poly_matrix[k-1,:], (1,)) .- poly_matrix[k-2,:]
    end
    return poly_matrix
end

##Chebyshev basis of the form [T_0(x), T_1(x), ..., T_n(x)] where T is the Chebyshev polynomial of the first kind
struct ChebyshevPolyBasis{Int, AbstractArray} <: TensorProductBasis
    n::Int
    poly_matrix::AbstractArray
    function ChebyshevPolyBasis(n)
        poly_matrix = create_legendre_poly(n)
        new{typeof(n), typeof(poly_matrix)}(n, poly_matrix)
    end
end

function (basis::ChebyshevPolyBasis)(x)
    return [evalpoly(x, basis.poly_matrix[i,:]) for i in 1:basis.n]
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

function create_legendre_poly(n)
    poly_matrix = zeros(n, n)
    poly_matrix[1, 1] = 1
    poly_matrix[2, 2] = 1
    for k in 3:n
        poly_matrix[k,:] = ((2*k-3)/(k-1)).*circshift(poly_matrix[k-1,:], (1,)) .- ((k-2)/(k-1)).*poly_matrix[k-2,:]
    end
    return poly_matrix
end

##Legendre basis of the form [P_0(x), P_1(x), ..., P_n(x)]
struct LegendrePolyBasis{Int, AbstractArray} <: TensorProductBasis
    n::Int
    poly_matrix::AbstractArray
    function LegendrePolyBasis(n)
        poly_matrix = create_legendre_poly(n)
        new{typeof(n), typeof(poly_matrix)}(n, poly_matrix)
    end
end

function (basis::LegendrePolyBasis)(x)
    return [evalpoly(x, basis.poly_matrix[i,:]) for i in 1:basis.n]
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
