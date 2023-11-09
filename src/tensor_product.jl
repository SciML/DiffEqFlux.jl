abstract type TensorProductBasis <: Function end

@concrete struct TensorProductBasisFunction
    f
    n
end

(basis::TensorProductBasisFunction)(x) = map(i -> basis.f(i, x), 1:(basis.n))

"""
    ChebyshevBasis(n)

Constructs a Chebyshev basis of the form [T_{0}(x), T_{1}(x), ..., T_{n-1}(x)] where T_j(.)
is the j-th Chebyshev polynomial of the first kind.

Arguments:

- `n`: number of terms in the polynomial expansion.
"""
ChebyshevBasis(n) = TensorProductBasisFunction(__chebyshev, n)

__chebyshev(i, x) = cos(i * acos(x))

"""
    SinBasis(n)

Constructs a sine basis of the form [sin(x), sin(2*x), ..., sin(n*x)].

Arguments:

- `n`: number of terms in the sine expansion.
"""
SinBasis(n) = TensorProductBasisFunction(sin ∘ *, n)

"""
    CosBasis(n)

Constructs a cosine basis of the form [cos(x), cos(2*x), ..., cos(n*x)].

Arguments:

- `n`: number of terms in the cosine expansion.
"""
CosBasis(n) = TensorProductBasisFunction(cos ∘ *, n)

"""
    FourierBasis(n)

Constructs a Fourier basis of the form
F_j(x) = j is even ? cos((j÷2)*x) : sin((j÷2)*x) => [F_0(x), F_1(x), ..., F_n(x)].

Arguments:

- `n`: number of terms in the Fourier expansion.
"""
FourierBasis(n) = TensorProductBasisFunction(__fourier, n)

__fourier(i::Int, x) = ifelse(iseven(i), cos(i * x / 2), sin(i * x / 2))

"""
    LegendreBasis(n)

Constructs a Legendre basis of the form [P_{0}(x), P_{1}(x), ..., P_{n-1}(x)] where
P_j(.) is the j-th Legendre polynomial.

Arguments:

- `n`: number of terms in the polynomial expansion.
"""
LegendreBasis(n) = TensorProductBasisFunction(__legendre_poly, n)

## Source: https://github.com/ranocha/PolynomialBases.jl/blob/master/src/legendre.jl
function __legendre_poly(i::Int, x)
    p = i - 1
    a = one(x)
    b = x

    p ≤ 0 && return a
    p == 1 && return b

    for j in 2:p
        a, b = b, ((2j - 1) * x * b - (j - 1) * a) / j
    end

    return b
end

"""
    PolynomialBasis(n)

Constructs a Polynomial basis of the form [1, x, ..., x^(n-1)].

Arguments:

- `n`: number of terms in the polynomial expansion.
"""
PolynomialBasis(n) = TensorProductBasisFunction(__polynomial, n)

__polynomial(i, x) = x^(i - 1)

"""
    TensorLayer(model, out_dim::Int, init_p::F = randn) where {F <: Function}

Constructs the Tensor Product Layer, which takes as input an array of n tensor
product basis, [B_1, B_2, ..., B_n] a data point x, computes
z[i] = W[i,:] ⨀ [B_1(x[1]) ⨂ B_2(x[2]) ⨂ ... ⨂ B_n(x[n])], where W is the layer's weight,
and returns [z[1], ..., z[out]].

Arguments:

- `model`: Array of TensorProductBasis [B_1(n_1), ..., B_k(n_k)], where k corresponds to the
  dimension of the input.
- `out`: Dimension of the output.
- `p`: Optional initialization of the layer's weight. Initialized to standard normal by
  default.
"""
function TensorLayer(model, out_dim::Int, init_p::F = randn) where {F <: Function}
    number_of_weights = prod(Base.Fix2(getproperty, :n), model)
    return Chain(x -> mapfoldl(((m, xᵢ),) -> m(xᵢ), kron, zip(model, x)),
        Dense(number_of_weights => out_dim; use_bias = false, init_weight = init_p))
end
