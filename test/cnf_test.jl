using OrdinaryDiffEq
println("Starting tests")
using Flux, DiffEqFlux
using Test
using Distributions
using Distances
using LinearAlgebra, Tracker, Zygote

##callback to be used by all tests
function cb(p,l)
    false
end

###
#test for default base distribution and monte_carlo = true
###

nn = Chain(Dense(1, 1, tanh))
data_train = [Float32(rand(Beta(7,7))) for i in 1:100]
tspan = (0.0,10.0)
ffjord_test_mc = FFJORD(nn,tspan,Tsit5(),monte_carlo=true)

function loss_adjoint(θ)
    logpx = [ffjord_test_mc(x,θ) for x in data_train]
    loss = -mean(logpx)
end

res = DiffEqFlux.sciml_train(loss_adjoint, ffjord_test_mc.p,
                                        ADAM(0.1), cb=cb,
                                        maxiters = 300)

θopt = res.minimizer
data_validate = [Float32(rand(Beta(7,7))) for i in 1:100]
actual_pdf = [pdf(Beta(7,7),r) for r in data_validate]
#use direct trace calculation for predictions
learned_pdf = [exp(ffjord_test_mc(r,θopt,false)) for r in data_validate]

@test totalvariation(learned_pdf, actual_pdf)/100 < 0.25

###
#test for alternative base distribution and monte_carlo = false
###

nn = Chain(Dense(1, 3, tanh), Dense(3, 1, tanh))
data_train = [Float32(rand(Normal(6.0,0.7))) for i in 1:100]
tspan = (0.0,10.0)
ffjord_test = FFJORD(nn,tspan,Tsit5(),base_dist=Normal(0,2))

function loss_adjoint(θ)
    logpx = [ffjord_test(x,θ) for x in data_train]
    loss = -mean(logpx)
end

res = DiffEqFlux.sciml_train(loss_adjoint, ffjord_test.p,
                                          ADAM(0.1), cb = cb,
                                          maxiters = 300)

θopt = res.minimizer
data_validate = [Float32(rand(Normal(6.0,0.7))) for i in 1:100]
actual_pdf = [pdf(Normal(6.0,0.7),r) for r in data_validate]
learned_pdf = [exp(ffjord_test(r, θopt)) for r in data_validate]

@test totalvariation(learned_pdf, actual_pdf)/100 < 0.25

###
#test for alternative multivariate distribution with monte_carlo = false
###

nn = Chain(Dense(2, 2, tanh))
μ = ones(2)
Σ = 7*I + zeros(2,2)
mv_normal = MvNormal(μ, Σ)
data_train = [Float32.(rand(mv_normal)) for i in 1:100]
tspan = (0.0,10.0)
ffjord_test = FFJORD(nn,tspan,Tsit5())

function loss_adjoint(θ)
    logpx = [ffjord_test(x,θ) for x in data_train]
    loss = -mean(logpx)
end

res = DiffEqFlux.sciml_train(loss_adjoint, ffjord_test.p,
                                          ADAM(0.1), cb = cb,
                                          maxiters = 300)

θopt = res.minimizer
data_validate = [Float32.(rand(mv_normal)) for i in 1:100]
actual_pdf = [pdf(mv_normal,r) for r in data_validate]
learned_pdf = [exp(ffjord_test(r, θopt)) for r in data_validate]

@test totalvariation(learned_pdf, actual_pdf)/100 < 0.25
