module DiffEqFlux

using GalacticOptim, DataInterpolations, DiffEqBase, DiffResults, DiffEqSensitivity,
      Distributions, ForwardDiff, Flux, Requires, Adapt, LinearAlgebra, RecursiveArrayTools,
      StaticArrays, Base.Iterators, Printf, Zygote

using DistributionsAD
import ProgressLogging, ZygoteRules
import ConsoleProgressMonitor, TerminalLoggers, LoggingExtras
import ArrayInterface

import Logging

gpu_or_cpu(x) = Array

function diffeq_fd(p,f,n,prob,solver=nothing,args...;u0=prob.u0,kwargs...)
  @warn("diffeq_fd has been deprecated in the update of DiffEqFlux to Zygote support. Use the solve function with sensealg=ForwardDiffSensitivity() to recover the same functionality. See https://docs.juliadiffeq.org/latest/analysis/sensitivity/ for more details")
  f(concrete_solve(prob,solver,u0,p,args...;kwargs...))
end

function diffeq_rd(p,prob,solver=nothing,args...;u0=prob.u0,kwargs...)
  @warn("diffeq_rd has been deprecated in the update of DiffEqFlux to Zygote support. Use the concrete_solve function with sensealg=TrackerAdjoint() to recover the same functionality. See https://docs.juliadiffeq.org/latest/analysis/sensitivity/ for more details")
  concrete_solve(prob,solver,u0,p,args...;kwargs...)
end

function diffeq_adjoint(p,prob,solver=nothing,args...;u0=prob.u0,kwargs...)
  @warn("diffeq_adjoint has been deprecated in the update of DiffEqFlux to Zygote support. Use the concrete_solve function to recover the same functionality. See https://docs.juliadiffeq.org/latest/analysis/sensitivity/ for more details")
  concrete_solve(prob,solver,u0,p,args...;kwargs...)
end

function neural_ode(args...;kwargs...)
  @error("neural_ode has be removed and replaced by NeuralODE. Please consult the README for more details.")
end

function neural_ode_rd(args...;kwargs...)
  @error("neural_ode_rd has be removed and replaced by NeuralODE. Please consult the README for more details.")
end

function neural_dmsde(args...;kwargs...)
  @error("neural_dmsde has be removed and replaced by NeuralDMSDE. Please consult the README for more details.")
end

function neural_dmsde(model,x,mp,tspan,
                      args...;kwargs...)
    error("neural_dmsde has been deprecated with the change to Zygote. Please see the documentation on the new NeuralDSDE layer.")
end

function neural_ode_rd(model,x,tspan,
                       args...;
                       kwargs...)
    error("neural_ode_rd has been deprecated with the change to Zygote. Please see the documentation on the new NeuralODE layer.")
end

function neural_ode(model,x,tspan,args...;kwargs...)
    error("neural_ode has been deprecated with the change to Zygote. Please see the documentation on the new NeuralODE layer.")
end

# ForwardDiff integration

ZygoteRules.@adjoint function ForwardDiff.Dual{T}(x, ẋ::Tuple) where T
  @assert length(ẋ) == 1
  ForwardDiff.Dual{T}(x, ẋ), ḋ -> (ḋ.partials[1], (ḋ.value,))
end

ZygoteRules.@adjoint ZygoteRules.literal_getproperty(d::ForwardDiff.Dual{T}, ::Val{:partials}) where T =
  d.partials, ṗ -> (ForwardDiff.Dual{T}(ṗ[1], 0),)

ZygoteRules.@adjoint ZygoteRules.literal_getproperty(d::ForwardDiff.Dual{T}, ::Val{:value}) where T =
  d.value, ẋ -> (ForwardDiff.Dual{T}(0, ẋ),)

ZygoteRules.@adjoint ZygoteRules.literal_getproperty(A::Tridiagonal, ::Val{:dl}) = A.dl,y -> Tridiagonal(dl,zeros(length(d)),zeros(length(du)),)
ZygoteRules.@adjoint ZygoteRules.literal_getproperty(A::Tridiagonal, ::Val{:d}) = A.d,y -> Tridiagonal(zeros(length(dl)),d,zeros(length(du)),)
ZygoteRules.@adjoint ZygoteRules.literal_getproperty(A::Tridiagonal, ::Val{:du}) = A.dl,y -> Tridiagonal(zeros(length(dl)),zeros(length(d),du),)
ZygoteRules.@adjoint Tridiagonal(dl, d, du) = Tridiagonal(dl, d, du), p̄ -> (diag(p̄[2:end,1:end-1]),diag(p̄),diag(p̄[1:end-1,2:end]))

include("ffjord.jl")
include("train.jl")
include("fast_layers.jl")
include("neural_de.jl")
include("require.jl")
include("spline_layer.jl")
include("tensor_product_basis.jl")
include("tensor_product_layer.jl")
include("collocation.jl")
include("hnn.jl")
include("multiple_shooting.jl")

export diffeq_fd, diffeq_rd, diffeq_adjoint
export DeterministicCNF, FFJORD, NeuralODE, NeuralDSDE, NeuralSDE, NeuralCDDE, NeuralDAE, NeuralODEMM, TensorLayer, AugmentedNDELayer, SplineLayer, NeuralHamiltonianDE
export HamiltonianNN
export ChebyshevBasis, SinBasis, CosBasis, FourierBasis, LegendreBasis, PolynomialBasis
export neural_ode, neural_ode_rd
export neural_dmsde
export FastDense, StaticDense, FastChain, initial_params

export EpanechnikovKernel, UniformKernel, TriangularKernel, QuarticKernel
export TriweightKernel, TricubeKernel, GaussianKernel, CosineKernel
export LogisticKernel, SigmoidKernel, SilvermanKernel
export collocate_data

export multiple_shoot

end
