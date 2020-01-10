using Flux, DiffEqFlux, OrdinaryDiffEq, LinearAlgebra, Tracker, Zygote, Test

GDP = [11394358246872.6, 11886411296037.9, 12547852149499.6, 13201781525927, 14081902622923.3, 14866223429278.3, 15728198883149.2, 16421593575529.9, 17437921118338, 18504710349537.1, 19191754995907.1, 20025063402734.2, 21171619915190.4, 22549236163304.4, 22999815176366.2, 23138798276196.2, 24359046058098.6, 25317009721600.9, 26301669369287.8, 27386035164588.8, 27907493159394.4, 28445139283067.1, 28565588996657.6, 29255060755937.6, 30574152048605.8, 31710451102539.4, 32786657119472.8, 34001004119223.5, 35570841010027.7, 36878317437617.5, 37952345258555.4, 38490918890678.7, 39171116855465.5, 39772082901255.8, 40969517920094.4, 42210614326789.4, 43638265675924.6, 45254805649669.6, 46411399944618.2, 47929948653387.3, 50036361141742.2, 51009550274808.6, 52127765360545.5, 53644090247696.9, 55995239099025.6, 58161311618934.2, 60681422072544.7, 63240595965946.1, 64413060738139.7, 63326658023605.9, 66036918504601.7, 68100669928597.9, 69811348331640.1, 71662400667935.7, 73698404958519.1, 75802901433146, 77752106717302.4, 80209237761564.8, 82643194654568.3]
function monomial(cGDP, parameters, t)
    α1, β1, nu1, nu2, δ, δ2 = parameters

    [α1 * ((cGDP[1]))^β1]
end
function monomial(cGDP::Tracker.TrackedArray, parameters, t)
    α1, β1, nu1, nu2, δ, δ2 = parameters

    Tracker.collect([α1 * ((cGDP[1]))^β1])
end

GDP0 = GDP[1]

tspan = (1.0, 59.0)
p = [474.8501513113645, 0.7036417845990167, 0.0, 1e-10, 1e-10, 1e-10]
u0 = [GDP0]
if false
    prob = ODEProblem(monomial,[GDP0],tspan,p)
else ## false crashes. that is when i am tracking the initial conditions
    prob = ODEProblem(monomial,u0,tspan,p)
end
function predict_rd() # Our 1-layer neural network
  #diffeq_rd(p,prob,Tsit5(),saveat=1.0:1.0:59.0)
  Array(concrete_solve(prob,Tsit5(),prob.u0,p,saveat=1.0:1.0:59.0,reltol=1e-4,sensealg=TrackerAdjoint()))
end

function loss_rd() ##L2 norm biases the newer times unfairly
    ##Graph looks better if we minimize relative error squared
    c = 0.0
    a = predict_rd()
    d = 0.0
    for i=1:59
        c += (a[i][1]/GDP[i]-1)^2 ## L2 of relative error
    end
    c + 3 * d
end

data = Iterators.repeated((), 100)
opt = ADAM(0.01)

peek = function () #callback function to observe training
    #reduces training speed by a lot
    println("Loss: ",loss_rd())
end

peek()
Flux.train!(loss_rd, Flux.params(p,u0), data, opt, cb=peek)
peek()

@test loss_rd() < 0.2

function monomial(dcGDP, cGDP, parameters, t)
    α1, β1, nu1, nu2, δ, δ2 = parameters

    dcGDP[1] = α1 * ((cGDP[1]))^β1
end


GDP0 = GDP[1]

tspan = (1.0, 59.0)
p = [474.8501513113645, 0.7036417845990167, 0.0, 1e-10, 1e-10, 1e-10]
u0 = [GDP0]
if false
    prob = ODEProblem(monomial,[GDP0],tspan,p)
else ## false crashes. that is when i am tracking the initial conditions
    prob = ODEProblem(monomial,u0,tspan,p)
end
function predict_adjoint() # Our 1-layer neural network
  Array(concrete_solve(prob,Tsit5(),prob.u0,p,saveat=1.0,reltol=1e-4))
end

function loss_adjoint() ##L2 norm biases the newer times unfairly
    ##Graph looks better if we minimize relative error squared
    c = 0.0
    a = predict_adjoint()
    d = 0.0
    for i=1:59
        c += (a[i][1]/GDP[i]-1)^2 ## L2 of relative error
    end
    c + 3 * d
end

data = Iterators.repeated((), 100)
opt = ADAM(0.01)

peek = function () #callback function to observe training
    #reduces training speed by a lot
    println("Loss: ",loss_adjoint())
end

peek()
Flux.train!(loss_adjoint, Flux.params(p,u0), data, opt, cb=peek)
peek()
@test loss_adjoint() < 0.2
