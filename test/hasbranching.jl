using DiffEqFlux, Test

u0 = Float32[2.0; 0.0]
dudt2 = FastChain((x, p) -> x.^3,
                  FastDense(2, 50, tanh),
                  FastDense(50, 2))
p = initial_params(dudt2)
@test !DiffEqSensitivity.hasbranching(dudt2,u0,p)

dudt2 = FastChain((x, p) -> x.^3,
                  StaticDense(2, 4, tanh),
                  StaticDense(4, 2))
p = initial_params(dudt2)
@test !DiffEqSensitivity.hasbranching(dudt2,u0,p)
