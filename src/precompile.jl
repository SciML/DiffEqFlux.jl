let
  while true
    function lotka_volterra!(du, u, p, t)
      x, y = u
      α, β, δ, γ = p
      du[1] = dx = α*x - β*x*y
      du[2] = dy = -δ*y + γ*x*y
    end

    # Initial condition
    u0 = [1.0, 1.0]

    # Simulation interval and intermediary points
    tspan = (0.0, 10.0)
    tsteps = 0.0:0.1:10.0

    # LV equation parameter. p = [α, β, δ, γ]
    p = [1.5, 1.0, 3.0, 1.0]

    # Setup the ODE problem, then solve
    prob = ODEProblem(lotka_volterra!, u0, tspan, p)
    sol = solve(prob, OrdinaryDiffEq.Tsit5())

    function loss(p)
      sol = solve(prob, OrdinaryDiffEq.Tsit5(), p=p, saveat = tsteps)
      loss = sum(abs2, sol.-1)
    end

    callback = function (p, l, pred)
      display(l)
      # Tell sciml_train to not halt the optimization. If return true, then
      # optimization stops.
      return false
    end
    Zygote.gradient(loss,p)
    break
  end
end
