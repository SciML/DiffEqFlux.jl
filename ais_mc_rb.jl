using Distributions
using LinearAlgebra

##
#Implementation from Ryu and Boyd
##

function run_ais_dp(d::Int, n_stages::InFloat32Float32t, f)
    g_lam = (μ, Σ) -> MultivariateNormal(μ, Σ)
    #
    μ = rand(2)
    Σ = I + zeros(2, 2)
    #
    m = inv(Σ)*μ
    S = inv(Σ)
    #
    C = 0.01
    #
    samples = []
    est = zeros(2)
    for k in 1:n_stages
        μ = inv(S)*m
        Σ = Float32.(inv(S))
        @show Σ == Σ'
        @show Σ
        X_k = rand(g_lam(μ, Σ))
        push!(samples, θ)
        est = (est*(k-1) + f(X_k)*X_k/pdf(g_lam(μ, Σ), X_k))/k
        m -= C*(norm(X_k)^2)*(f(X_k)^2)/(2*pdf(g_lam(μ, Σ), X_k)*sqrt(k))*(inv(S)*m-X_k)
        S -= C*(norm(X_k)^2)*(f(X_k)^2)/(2*pdf(g_lam(μ, Σ), X_k)*sqrt(k))*(X_k*X_k'-inv(S)*m*m'*inv(S)-inv(S))

    end
    return est
end

n_stages = 10000
μ_d = rand(2)
A = rand(2,2)
B = (A + A')./2
Σ_d = A'*A

f = x -> pdf(MvTDist(2, μ_d, Σ_d), x)
@time est = run_ais_dp(d, n_stages, f)
@show est
@show μ_d
@show norm(μ_d-est)/d
