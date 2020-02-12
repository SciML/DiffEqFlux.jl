using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots

u0 = Float32[2.; 0.]
datasize = 30
tspan = (0.0f0,1.5f0)

function trueODEfunc(du,u,p,t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end

t = range(tspan[1],tspan[2],length=datasize)
prob = ODEProblem(trueODEfunc,u0,tspan)
sol = solve(prob,Tsit5(),saveat=t)
training_data = [(sol.u[n], sol.u[n+1]) for n in 1:length(sol)-1]

dudt2 = FastChain((x,p) -> x.^3,
            FastDense(2,50,tanh),
            FastDense(50,2))

dt = t[2]-t[1]
n_ode_tspan = (0.0f0, dt)
n_ode = NeuralODE(dudt2,n_ode_tspan,Tsit5(),saveat=[dt])

function predict_n_ode(u, p)
  n_ode(u,p)
end

I = Iterators.Stateful(Iterators.cycle(1:length(training_data)))

function loss_n_ode(p)
    i = popfirst!(I)
    u_i = training_data[i][1]
    u_next = training_data[i][2]
    pred = predict_n_ode(u_i,p)
    loss = sum(abs2,u_next .- pred)
    loss,pred
end

loss_n_ode(n_ode.p) # n_ode.p stores the initial parameters of the neural ODE

cb = function (p,l,pred) #callback function to observe training
  display(l)
  return false
end

# Display the ODE with the initial parameter values.
cb(n_ode.p,loss_n_ode(n_ode.p)...)

res1 = DiffEqFlux.sciml_train(loss_n_ode, n_ode.p, ADAM(0.05), cb = cb, maxiters = 300)
cb(res1.minimizer,loss_n_ode(res1.minimizer)...)

res2 = DiffEqFlux.sciml_train(loss_n_ode, res1.minimizer, LBFGS(), cb = cb)
cb(res2.minimizer,loss_n_ode(res2.minimizer)...)

