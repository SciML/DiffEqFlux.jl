"""
Constructs a Liquid time-constant Networks [1].

References:
[1] Hasani, R., Lechner, M., Amini, A., Rus, D. & Grosu, R. Liquid time-constant
networks. 2020.
"""

struct LTCCell{F,A,V,S,AB,OU,TA}
    σ::F
    Wi::A
    Wh::A
    b::V
    A::AB
    τ::TA
    _ode_unfolds::OU
    state0::S
    elapsed_time
end

LTCCell(in::Integer, out::Integer, σ=tanh; init=Flux.glorot_uniform, initb=zeros, init_state=zeros, init_tau=rand, ode_unfolds=6, elapsed_time=1.0) =
    LTCCell(σ, init(out, in), init(out, out), initb(out), initb(out), init_tau(out), ode_unfolds, init_state(out,1), elapsed_time)

Flux.trainable(m::LTCCell) = (m.Wi, m.Wh, m.b, m.A, m.τ,)

function (m::LTCCell)(h, x)
  h = _ode_solver(m::LTCCell, h, x)
  out = h
  return h, out
end

function _ode_solver(m::LTCCell, h, x)
    σ, Wi, Wh, b, τ, A = m.σ, m.Wi, m.Wh, m.b, m.τ, m.A # assert it is > 0
    τ = Flux.softplus.(τ) # to ensure τ>=0
    Δt = m.elapsed_time/m._ode_unfolds
    for t = 1:m._ode_unfolds # FuseStep
        f = σ.(Wi*x .+ Wh*h .+ b)
        numerator = h .+ Δt .* f .* A
        denominator = 1 .+ Δt .* (1 ./ τ .+ f)
        h = numerator ./ (denominator .+ 1e-8) # insert epsilon
        h = clamp.(h, -1, 1) # to ensure stability
    end
    return h
end

LTC(a...; ka...) = Flux.Recur(LTCCell(a...; ka...))
Flux.Recur(m::LTCCell) = Flux.Recur(m, m.state0)
