module DiffEqFluxOrdinaryDiffEqTsit5Ext

using DiffEqFlux: DiffEqFlux
using OrdinaryDiffEqTsit5: Tsit5ConstantCache
import OrdinaryDiffEqTsit5

function DiffEqFlux.local_regularization_step(
        integrator, cache::Tsit5ConstantCache, p
    )
    step_data = OrdinaryDiffEqTsit5.perform_step(integrator, cache, p)
    residuals = step_data.utilde ./ (
        integrator.opts.abstol .+
        max.(abs.(integrator.uprev), abs.(step_data.u)) .* integrator.opts.reltol
    )
    reg_val = sqrt(sum(abs2, residuals) / length(step_data.u)) * step_data.dt
    return reg_val, step_data.nf + DiffEqFlux.__ode_nfe(integrator.sol)
end

end
