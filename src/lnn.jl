"""
Constructs a Lagrangian Neural Network [1].

References:
[1] Miles Cranmer, Sam Greydanus, Stephan Hoyer, Peter Battaglia, David Spergel, and Shirley Ho.Lagrangian Neural Networks.
    InICLR 2020 Workshop on Integration of Deep Neural Modelsand Differential Equations, 2020.
"""

struct LagrangianNN
    model
    re
    params

    # Define inner constructor method
    function LagrangianNN(model; p = nothing)
        _p, re = Flux.destructure(model)
        if p === nothing
            p = _p
        end
        return new(model, re, p)
    end
end

function (nn::LagrangianNN)(x, p = nn.params)
    @assert size(x,1) % 2 === 0 # velocity df should be equal to coords degree of freedom
    M = div(size(x,1), 2) # number of velocities degrees of freedom
    re = nn.re
    hess = x -> Zygote.hessian_reverse(x->sum(re(p)(x)), x) # we have to compute the whole hessian
    hess = hess(x)[M+1:end, M+1:end]  # takes only velocities
    inv_hess = GenericLinearAlgebra.pinv(hess)

    _grad_q = x -> Zygote.gradient(x->sum(re(p)(x)), x)[end]
    _grad_q = _grad_q(x)[1:M,:] # take only coord derivatives
    out1 =_grad_q

    # Second term
    _grad_qv = x -> Zygote.gradient(x->sum(re(p)(x)), x)[end]
    _jac_qv = x -> Zygote.jacobian(x->_grad_qv(x), x)[end]
    out2 = _jac_qv(x)[1:M,M+1:end] * x[M+1:end] # take only dqdq_dot derivatives

    return inv_hess * (out1 .+ out2)
end
