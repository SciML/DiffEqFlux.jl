"""
Returns the a total loss after trying a 'Direct multiple shooting' on ODE data
and an array of predictions from the each of the groups (smaller intervals).
In Direct Multiple Shooting, the Neural Network divides the interval into smaller intervals
and solves for them separately.
The default continuity term is 100 implying any losses arising from the non-continuity
of 2 different groups will be scaled by 100.

```julia
multiple_shoot(p, ode_data, tsteps, prob, loss_function, group_size; continuity_term=100)
```

Arguments:
  - `p`: The parameters of the Neural Network to be trained.
  - `ode_data`: Original Data to be modelled.
  - `tsteps`: Timesteps on which ode_data was calculated.
  - `prob`: ODE problem that the Neural Network attempts to solve.
  - `loss_function`: Any arbitrary function to calculate loss.
  - `group_size`: The group size achieved after splitting the ode_data into equal sizes.
  - `continuity_term`: Multiplying factor to ensure continuity of predictions throughout
    different groups.

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
    solver::DiffEqBase.AbstractODEAlgorithm,
    group_size::Integer;
    continuity_term::Real=100,
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
            ),
        ) for rg in ranges
    ]

    # Calculate multiple shooting loss
    loss = 0
    for (i, rg) in enumerate(ranges)
        y = ode_data[:, rg]
        ŷ = group_predictions[i]
        loss += loss_function(y, ŷ)

        if i > 1
            # Ensure continuity between last state in previous prediction
            # and current initial condition in ode_data
            loss += continuity_term * sum(abs, y[:, 1] - group_predictions[i - 1][:, end])
        end
    end

    return loss, group_predictions
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
