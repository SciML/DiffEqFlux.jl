module DiffEqFluxOrdinaryDiffEqTsit5Ext

import DiffEqFlux: local_regularization_step
using SciMLBase: DEIntegrator
import OrdinaryDiffEqTsit5: Tsit5, perform_step

function local_regularization_step(integrator::DEIntegrator{Alg}, p) where {Alg <: Tsit5}
    _, reg_val, local_nf, _ = perform_step(integrator, p)
    reg_val, local_nf
end

end

