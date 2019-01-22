using OrdinaryDiffEq, StochasticDiffEq, Flux, DiffEqFlux

x = Float32[2.; 0.]
tspan = (0.0f0,25.0f0)
dudt = Chain(Dense(2,50,tanh),Dense(50,2))

neural_ode(dudt,x,tspan,Tsit5(),save_everystep=false,save_start=false)
neural_ode(dudt,x,tspan,Tsit5(),saveat=0.1)

neural_msde(dudt,x,[0.1,0.1],tspan,SOSRI(),saveat=0.1)
