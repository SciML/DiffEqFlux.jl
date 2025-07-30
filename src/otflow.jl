struct OTFlow <: AbstractLuxLayer
	d::Int
	m::Int
	r::Int
end

OTFlow(d::Int, m::Int; r::Int = min(10, d)) = OTFlow(d, m, r)

function Lux.initialparameters(rng::AbstractRNG, l::OTFlow)
	w = randn(rng, Float32, l.m) .* 0.01f0
	A = randn(rng, Float32, l.r, l.d + 1) .* 0.01f0
	b = zeros(Float32, l.d + 1)
	c = randn(rng, Float32, l.m) .* 0.01f0
	K0 = randn(rng, Float32, l.m, l.d + 1) .* 0.01f0
	K1 = randn(rng, Float32, l.m, l.m) .* 0.01f0
	b0 = zeros(Float32, l.m)
	b1 = zeros(Float32, l.m)
	return (; w, A, b, c, K0, K1, b0, b1)
end

sigma(x) = log(exp(x) + exp(-x))
sigma_prime(x) = tanh(x)
sigma_double_prime(x) = 1 - tanh(x)^2

function resnet_forward(x::AbstractVector, t::Real, ps)
	s = vcat(x, t)
	u0 = sigma.(ps.K0 * s .+ ps.b0)
	u1 = u0 .+ sigma.(ps.K1 * u0 .+ ps.b1)
	return u1
end

function potential(x::AbstractVector, t::Real, ps)
	s = vcat(x, t)
	N = resnet_forward(x, t, ps)
	quadratic_term = 0.5 * s' * (ps.A' * ps.A) * s
	linear_term = ps.b' * s
	neural_term = sum((ps.w .+ ps.c) .* N)
	return neural_term + quadratic_term + linear_term
end

function gradient(x::AbstractVector, t::Real, ps, d::Int)
	s = vcat(x, t)
	u0 = sigma.(ps.K0 * s .+ ps.b0)
	z1 = (ps.w .+ ps.c) .+ ps.K1' * (sigma_prime.(ps.K1 * u0 .+ ps.b1) .* (ps.w .+ ps.c))
	z0 = ps.K0' * (sigma_prime.(ps.K0 * s .+ ps.b0) .* z1)
	grad = z0 + (ps.A' * ps.A) * s + ps.b
	return grad[1:d]
end

function trace(x::AbstractVector, t::Real, ps, d::Int)
	s = vcat(x, t)
	u0 = sigma.(ps.K0 * s .+ ps.b0)
	z1 = (ps.w .+ ps.c) .+ ps.K1' * (sigma_prime.(ps.K1 * u0 .+ ps.b1) .* (ps.w .+ ps.c))
	K0_E = ps.K0[:, 1:d]
	A_E = ps.A[:, 1:d]
	t0 = sum(sigma_double_prime.(ps.K0 * s .+ ps.b0) .* z1 .* (K0_E .^ 2))
	J = Diagonal(sigma_prime.(ps.K0 * s .+ ps.b0)) * K0_E
	t1 = sum(sigma_double_prime.(ps.K1 * u0 .+ ps.b1) .* (ps.w .+ ps.c) .* (ps.K1 * J) .^ 2)
	trace_A = tr(A_E' * A_E)
	return t0 + t1 + trace_A
end

function (l::OTFlow)(xt::Tuple{AbstractVector, Real}, ps, st)
	x, t = xt
	v = -gradient(x, t, ps, l.d)
	tr = -trace(x, t, ps, l.d)
	return (v, tr), st
end

function simple_loss(x::AbstractVector, t::Real, l::OTFlow, ps)
	(v, tr), _ = l((x, t), ps, NamedTuple())
	return sum(v .^ 2) / 2 - tr
end

function manual_gradient(x::AbstractVector, t::Real, l::OTFlow, ps)
	s = vcat(x, t)
	u0 = sigma.(ps.K0 * s .+ ps.b0)
	u1 = u0 .+ sigma.(ps.K1 * u0 .+ ps.b1)
	v = -gradient(x, t, ps, l.d)
	tr = -trace(x, t, ps, l.d)
	grad_w = u1
	grad_c = u1
	grad_A = (ps.A * s) * s'
	grad_b = similar(ps.b)
	return (w = grad_w, A = grad_A, b = grad_b, c = grad_c,
		K0 = zeros(l.m, l.d + 1), K1 = zeros(l.m, l.m),
		b0 = zeros(l.m), b1 = zeros(l.m))
end
