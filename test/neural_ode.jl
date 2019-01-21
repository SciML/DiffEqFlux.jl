using OrdinaryDiffEq, Flux, DiffEqFlux

x = Float32[2.; 0.]
tspan = (0.0f0,25.0f0)
dudt = Chain(Dense(2,50,tanh),Dense(50,2))

neural_ode(x,dudt,tspan,diffeq_adjoint,Tsit5(),saveat=0.1)
neural_ode(x,dudt,tspan,diffeq_fd,Tsit5(),saveat=0.1)
neural_ode(x,dudt,tspan,diffeq_rd,Tsit5(),saveat=0.1)
