println("Starting precompilation")
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
#test for default base distribution and deterministic trace CNF
###

nn = Chain(Dense(1, 1, tanh))
data_train = [Float32(rand(Beta(7,7))) for i in 1:100]
tspan = (0.0,10.0)
cnf_test = DeterministicCNFLayer(nn,tspan,Tsit5())

function loss_adjoint(θ)
    logpx = [cnf_test(x,θ) for x in data_train]
    loss = -mean(logpx)
end

res = DiffEqFlux.sciml_train(loss_adjoint, cnf_test.p,
                                        ADAM(0.1), cb=cb,
                                        maxiters = 300)

θopt = res.minimizer
data_validate = [Float32(rand(Beta(7,7))) for i in 1:100]
actual_pdf = [pdf(Beta(7,7),r) for r in data_validate]
#use direct trace calculation for predictions
learned_pdf = [exp(cnf_test(r,θopt)) for r in data_validate]

@test totalvariation(learned_pdf, actual_pdf)/100 < 0.25

###
#test for alternative base distribution and deterministic trace CNF
###

nn = Chain(Dense(1, 3, tanh), Dense(3, 1, tanh))
data_train = [Float32(rand(Normal(6.0,0.7))) for i in 1:100]
tspan = (0.0,10.0)
cnf_test = DeterministicCNFLayer(nn,tspan,Tsit5(),basedist=MvNormal([0.0],[2.0]))

function loss_adjoint(θ)
    logpx = [cnf_test(x,θ) for x in data_train]
    loss = -mean(logpx)
end

res = DiffEqFlux.sciml_train(loss_adjoint, cnf_test.p,
                                          ADAM(0.1), cb = cb,
                                          maxiters = 300)

θopt = res.minimizer
data_validate = [Float32(rand(Normal(6.0,0.7))) for i in 1:100]
actual_pdf = [pdf(Normal(6.0,0.7),r) for r in data_validate]
learned_pdf = [exp(cnf_test(r, θopt)) for r in data_validate]

@test totalvariation(learned_pdf, actual_pdf)/100 < 0.25

###
#test for multivariate distribution and deterministic trace CNF
###

nn = Chain(Dense(2, 2, tanh))
μ = ones(2)
Σ = 7*I + zeros(2,2)
mv_normal = MvNormal(μ, Σ)
data_train = [Float32.(rand(mv_normal)) for i in 1:100]
tspan = (0.0,10.0)
cnf_test = DeterministicCNFLayer(nn,tspan,Tsit5())

function loss_adjoint(θ)
    logpx = [ffjord_test(x,θ) for x in data_train]
    loss = -mean(logpx)
end

res = DiffEqFlux.sciml_train(loss_adjoint, cnf_test.p,
                                          ADAM(0.1), cb = cb,
                                          maxiters = 300)

θopt = res.minimizer
data_validate = [Float32.(rand(mv_normal)) for i in 1:100]
actual_pdf = [pdf(mv_normal,r) for r in data_validate]
learned_pdf = [exp(cnf_test(r, θopt)) for r in data_validate]

@test totalvariation(learned_pdf, actual_pdf)/100 < 0.25

###
#test for default multivariate distribution and FFJORD with regularizers
###

nn = Chain(Dense(1, 1, tanh))
data_train = [Float32(rand(Beta(7,7))) for i in 1:100]
tspan = (0.0,10.0)
ffjord_test = FFJORDLayer(nn,tspan,Tsit5())

function loss_adjoint(θ)
    logpx = [ffjord_test(x,θ)[1] for x in data_train]
    reg1 = [ffjord_test(x,θ)[2] for x in data_train]
    reg2 = [ffjord_test(x,θ)[3] for x in data_train]
    loss = -mean(logpx) + 0.01*mean(reg1) + 0.01*mean(reg2)
end

res = DiffEqFlux.sciml_train(loss_adjoint, ffjord_test.p,
                                        ADAM(0.01), cb=cb,
                                        maxiters = 50)


θopt = res.minimizer
data_validate = [Float32(rand(Beta(7,7))) for i in 1:100]
actual_pdf = [pdf(Beta(7,7),r) for r in data_validate]
#use direct trace calculation for predictions
learned_pdf = [exp(ffjord_test(r,θopt)[1]) for r in data_validate]

@test totalvariation(learned_pdf, actual_pdf)/100 < 0.25
