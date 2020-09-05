using DiffEqFlux, Flux, Test, OrdinaryDiffEq, CUDA
using Statistics
#= using Plots =#

CUDA.allowscalar(false)

## True Solution
u0 = [2.; 0.] |> gpu
datasize = 30
tspan = (0f0,25f0)
const true_A = cu([-0.1 2.0; -2.0 -0.1])

function trueODEfunc(du,u,p,t)
    du .= ((u.^3)'true_A)'
end

true_prob = ODEProblem(trueODEfunc, u0,tspan)

true_sol = solve(true_prob,BS3(),saveat=range(tspan[1],tspan[2],length=datasize))

#= true_sol_plot = solve(true_prob,Tsit5()) =#
#= plot(true_sol_plot) =#

## Neural ODE
dudt = Chain(Dense(2,50,tanh),Dense(50,2)) |> gpu

function ODEfunc(du,u,p,t)
    du .= dudt(u)
end

pred_prob = ODEProblem(ODEfunc, u0,tspan)
pred_sol = solve(pred_prob,BS3(),saveat=range(tspan[1],tspan[2],length=datasize))

## Loss
l1_loss(pred,target) = mean(abs.(pred-target))
l1_loss(Array(pred_sol),Array(true_sol))
