module DiffEqFluxOrdinaryDiffEqTsit5Ext

import DiffEqFlux: error_estimate_regularization
import ChainRulesCore as CRC
using SciMLBase: DEIntegrator, step!
import SciMLBase
import OrdinaryDiffEqTsit5
import OrdinaryDiffEqTsit5: Tsit5

function _tsit5_error_estimate_regularization(
        k1, uprev, model, p, t, dt, abstol, reltol
    )
    tab = OrdinaryDiffEqTsit5.Tsit5ConstantCacheActual(eltype(uprev), typeof(one(t)))
    (;
        c1, c2, c3, c4, a21, a31, a32, a41, a42, a43, a51, a52, a53, a54,
        a61, a62, a63, a64, a65, a71, a72, a73, a74, a75, a76,
        btilde1, btilde2, btilde3, btilde4, btilde5, btilde6, btilde7,
    ) = tab

    k2 = model(((@. uprev + dt * a21 * k1), t + c1 * dt), p)
    k3 = model(
        ((@. uprev + dt * (a31 * k1 + a32 * k2)), t + c2 * dt), p
    )
    k4 = model(
        ((@. uprev + dt * (a41 * k1 + a42 * k2 + a43 * k3)), t + c3 * dt), p
    )
    k5 = model(
        ((@. uprev + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)),
            t + c4 * dt),
        p,
    )
    g6 = @. uprev + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5)
    k6 = model((g6, t + dt), p)
    unew = @. uprev +
        dt * (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6)
    k7 = model((unew, t + dt), p)
    utilde = @. dt * (
        btilde1 * k1 + btilde2 * k2 + btilde3 * k3 + btilde4 * k4 +
            btilde5 * k5 + btilde6 * k6 + btilde7 * k7
    )
    residuals = @. utilde / (abstol + max(abs(uprev), abs(unew)) * reltol)
    return sqrt(sum(abs2, residuals) / length(unew)) * abs(dt)
end

function error_estimate_regularization(integrator::DEIntegrator{Alg}, model, p) where {Alg <: Tsit5}
    nf0 = CRC.@ignore_derivatives integrator.stats.nf
    integrator.p = p
    step!(integrator)
    reg_val = integrator.EEst * abs(integrator.t - integrator.tprev)
    local_nf = CRC.@ignore_derivatives integrator.stats.nf - nf0
    return reg_val, local_nf
end

function _tsit5_pullback_data(integrator)
    return CRC.@ignore_derivatives begin
        k1 = integrator.fsalfirst
        uprev = integrator.uprev
        (
            copy(k1),
            copy(uprev),
            integrator.t,
            integrator.dt,
            integrator.opts.abstol,
            integrator.opts.reltol,
        )
    end
end

function CRC.rrule(
        config::CRC.RuleConfig{>:CRC.HasReverseMode},
        ::typeof(error_estimate_regularization),
        integrator::DEIntegrator{Alg},
        model,
        p
    ) where {Alg <: Tsit5}
    pullback_data = _tsit5_pullback_data(integrator)
    y = error_estimate_regularization(integrator, model, p)

    function error_estimate_regularization_pullback(Delta)
        k1, uprev, t, dt, abstol, reltol = pullback_data
        _, back = CRC.rrule_via_ad(
            config,
            p -> _tsit5_error_estimate_regularization(
                k1, uprev, model, p, t, dt, abstol, reltol
            ),
            p,
        )
        d = back(first(Delta))
        dp = d[2]
        return CRC.NoTangent(), CRC.NoTangent(), CRC.NoTangent(), dp
    end
    return y, error_estimate_regularization_pullback
end

end
