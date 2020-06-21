using Distributions
using LinearAlgebra

##
#Implementation from Oh and Berger
##

function run_ais_dp(d::Int, n_samples::AbstractArray, n_stages::Int, f)
    g_lam = (λ, Σ) -> MvTDist(2, λ, Σ)
    Σ = I + zeros(d, d)
    λ = rand(d)
    ##array of functions
    w = []
    #array of functionals
    W = []
    #
    WΦ = zeros(n_stages, d+1)
    #array which contains an array of samples for each stage

    One = x -> 1
    Theta_components = [x -> x[k] for k in 1:d]

    N_k = 0
    for k in 1:n_stages
        w_k = θ -> f(θ)/pdf(g_lam(λ, Σ), θ)
        samples_k_list = []
        for n in 1:n_samples[k]
            θ = rand(g_lam(λ, Σ))
            push!(samples_k_list, θ)
        end
        W_k = h -> sum((h.(samples_k_list)).*(w_k.(samples_k_list)))
        WΦ[k, 1] = W_k(One)
        WΦ[k, 2:end] = [W_k(Theta_components[k-1]) for k in 2:d+1]
        sum_weights = sum(WΦ[:,1])
        A = [sum(WΦ[:,j])./sum_weights for j in 2:d+1]
        λ = (λ*N_k .+ WΦ[k,2:end])/(N_k + n_samples[k])
        N_k += n_samples[k]
    end
    return λ
end

n_stages = 1000
n_samples = [k for k in 1:n_stages]
d = 2
μ_d = rand(d)
A = rand(d,d)
B = (A + A')./2
Σ_d = A'*A

f = x -> pdf(MultivariateNormal(μ_d, Σ_d), x)
@time est = run_ais_dp(d, n_samples, n_stages, f)
@show est
@show μ_d
@show norm(μ_d-est)/d
