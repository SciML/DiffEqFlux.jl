"""
Returns the a total loss after trying a 'Direct multiple shooting' on ODE data
and an array of predictions from the each of the groups (smaller intervals).
In Direct Multiple Shooting, the Neural Network divides the interval into smaller intervals
and solves for them separately.
The default continuity term is 100, implying any losses arising from the non-continuity
of 2 different groups will be scaled by 100.

```julia
multiple_shoot(p, ode_data, tsteps, prob, loss_function, solver, group_size;
               continuity_term=100, sensealg=DiffEqSensitivity.InterpolatingAdjoint())
multiple_shoot(p, ode_data, tsteps, prob, loss_function, continuity_loss, solver, group_size;
               continuity_term=100, sensealg=DiffEqSensitivity.InterpolatingAdjoint())
```

Arguments:
  - `p`: The parameters of the Neural Network to be trained.
  - `ode_data`: Original Data to be modelled.
  - `tsteps`: Timesteps on which ode_data was calculated.
  - `prob`: ODE problem that the Neural Network attempts to solve.
  - `loss_function`: Any arbitrary function to calculate loss.
  - `continuity_loss`: Function that takes states ``\\hat{u}_{end}`` of group ``k`` and
  ``u_{0}`` of group ``k+1`` as input and calculates prediction continuity loss
  between them.
  If no custom `continuity_loss` is specified, `sum(abs, û_end - u_0)` is used.
  - `solver`: ODE Solver algorithm.
  - `group_size`: The group size achieved after splitting the ode_data into equal sizes.
  - `continuity_term`: Weight term to ensure continuity of predictions throughout
  different groups.
  - `sensealg`: Sensitivity algorithm, defaults to `InterpolatingAdjoint()`. Refer to
  the [Local Sensitivity Analysis](https://diffeq.sciml.ai/dev/analysis/sensitivity/)
  documentation for more details.

Note:
The parameter 'continuity_term' should be a relatively big number to enforce a large penalty
whenever the last point of any group doesn't coincide with the first point of next group.
"""
function multiple_shoot(
    p::AbstractArray,
    ode_data::AbstractArray,
    tsteps::AbstractArray,
    prob::ODEProblem,
    loss_function::Function,
    continuity_loss::Function,
    solver::DiffEqBase.AbstractODEAlgorithm,
    group_size::Integer;
    continuity_term::Real=100,
    sensealg::SciMLBase.AbstractSensitivityAlgorithm=InterpolatingAdjoint(),
)
    datasize = size(ode_data, 2)

    if group_size < 2 || group_size > datasize
        throw(DomainError(group_size, "group_size can't be < 2 or > number of data points"))
    end

    # Get ranges that partition data to groups of size group_size
    ranges = group_ranges(datasize, group_size)

    # Multiple shooting predictions
    group_predictions = [
        Array(
            solve(
                remake(
                    prob;
                    p=p,
                    tspan=(tsteps[first(rg)], tsteps[last(rg)]),
                    u0=ode_data[:, first(rg)],
                ),
                solver;
                saveat=tsteps[rg],
                sensealg=sensealg,
            ),
        ) for rg in ranges
    ]

    # Calculate multiple shooting loss
    loss = 0
    for (i, rg) in enumerate(ranges)
        u = ode_data[:, rg]
        û = group_predictions[i]
        loss += loss_function(u, û)

        if i > 1
            # Ensure continuity between last state in previous prediction
            # and current initial condition in ode_data
            loss +=
                continuity_term * continuity_loss(group_predictions[i - 1][:, end], u[:, 1])
        end
    end

    return loss, group_predictions
end

function multiple_shoot(
    p::AbstractArray,
    ode_data::AbstractArray,
    tsteps::AbstractArray,
    prob::ODEProblem,
    loss_function::Function,
    solver::DiffEqBase.AbstractODEAlgorithm,
    group_size::Integer;
    kwargs...,
)
    # Continuity loss between last state in previous prediction
    # and current initial condition in ode_data
    function continuity_loss(û_end, u_0)
        return sum(abs, û_end - u_0)
    end

    return multiple_shoot(
        p,
        ode_data,
        tsteps,
        prob,
        loss_function,
        continuity_loss,
        solver,
        group_size;
        kwargs...,
    )
end

"""
Get ranges that partition data of length `datasize` in groups of `groupsize` observations.
If the data isn't perfectly dividable by `groupsize`, the last group contains
the reminding observations.

```julia
group_ranges(datasize, groupsize)
```

Arguments:
- `datasize`: amount of data points to be partitioned
- `groupsize`: maximum amount of observations in each group

Example:
```julia-repl
julia> group_ranges(10, 5)
3-element Vector{UnitRange{Int64}}:
 1:5
 5:9
 9:10
```
"""
function group_ranges(datasize::Integer, groupsize::Integer)
    2 <= groupsize <= datasize || throw(
        DomainError(
            groupsize,
            "datasize must be positive and groupsize must to be within [2, datasize]",
        ),
    )
    return [i:min(datasize, i + groupsize - 1) for i in 1:groupsize-1:datasize-1]
end
