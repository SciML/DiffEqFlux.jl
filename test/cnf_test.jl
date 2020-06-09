using Distributions
using OrdinaryDiffEq
using Flux, DiffEqFlux
using Zygote, Tracker
using Test
using Distributions
using Distances

#test for default base distribution and monte_carlo = true
nn = Chain(Dense(1, 1, tanh))
data_train = [Float32(rand(Beta(7,7))) for i in 1:100]
tspan = (0.0,10.0)
ffjord_test_mc = FFJORD(nn,tspan,monte_carlo=true)

function loss_adjoint(θ)
    logpx = [ffjord_test_mc(x,θ) for x in data_train]
    loss = -mean(logpx)
end

res = DiffEqFlux.sciml_train(loss_adjoint, ffjord_test_mc.p,
                                        ADAM(0.1),
                                        maxiters = 100)

θopt = res.minimizer
data_validate = [Float32(rand(Beta(7,7))) for i in 1:100]
actual_pdf = [pdf(Beta(7,7),r) for r in data_validate]
#use direct trace calculation for predictions
learned_pdf = [exp(ffjord_test_mc(r,θopt,false)) for r in data_validate]

@test totalvariation(learned_pdf, actual_pdf)/100 < 0.3

#test for alternative base distribution and monte_carlo = false
nn = Chain(Dense(1, 1, tanh))
data_train = [Float32(rand(Normal(6.0,0.7))) for i in 1:100]
tspan = (0.0,10.0)
ffjord_test = FFJORD(nn,tspan,base_dist=Normal(0,2))

function loss_adjoint(θ)
    logpx = [ffjord_test(x,θ) for x in data_train]
    loss = -mean(logpx)
end

res = DiffEqFlux.sciml_train(loss_adjoint, ffjord_test.p,
                                          ADAM(0.1),
                                          maxiters = 100)

θopt = res.minimizer
data_validate = [Float32(rand(Normal(6.0,0.7))) for i in 1:100]
actual_pdf = [pdf(Normal(6.0,0.7),r) for r in data_validate]
learned_pdf = [exp(ffjord_test(r, θopt)) for r in data_validate]

@test totalvariation(learned_pdf, actual_pdf)/100 < 0.2
