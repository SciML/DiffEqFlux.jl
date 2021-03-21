"""
Returns the a total loss after trying a 'Direct multiple shooting' on ODE data, predictions on the whole ODE data and an array of predictions from the each of the groups (smaller intervals).
In Direct Multiple Shooting, the Neural Network divides the interval into smaller intervals and solves for them separately.
The default group size is 5 implying the whole dataset would be divided in groups of 5 and the Neural Network will solve on them individually.
The default continuity term is 100 implying any losses arising from the non-continuity of 2 different groups will be scaled by 100.

```julia
multiple_shoot(p,ode_data,tsteps,prob,loss_function,grp_size=5,continuity_strength=100)
```
Arguments:
- `p`: The parameters of the Neural Network to be trained.
- `ode_data`: Original Data to be modelled.
- `tsteps`: Timesteps on which ode_data was calculated.
- `prob`: ODE problem that the Neural Network attempts to solve.
- `loss_function`: Any arbitrary function to calculate loss.
- `grp_size`: The group size achieved after splitting the ode_data into equal sizes.
- `continuity_strength`: Multiplying factor to ensure continuity of predictions throughout different groups.

!!!note
The parameter 'continuity_strength' should be a relatively big number to enforce a large penalty whenever the last point of any group doesn't coincide with the first point of next group.
"""
function multiple_shoot(p :: Array, ode_data :: Array, tsteps, prob :: ODEProblem, loss_function ::Function, grp_size :: Integer = 5, continuity_term :: Integer = 100)
	datasize = length(ode_data[1,:])

	@assert (grp_size >= 1 && grp_size <= datasize) "grp_size can't be <= 1 or >= number of data points"

	tot_loss = 0

	if(grp_size == datasize)
		grp_predictions = [solve(remake(prob, p = p, tspan = (tsteps[1],tsteps[datasize]), u0 = ode_data[:,1]), Tsit5(),saveat = tsteps)]
		tot_loss = loss_function(ode_data, grp_predictions[1][:,1:grp_size])
		return tot_loss, grp_predictions[1], grp_predictions
	end

	if(grp_size == 1)
		# store individual single shooting predictions for each group
		grp_predictions = [solve(remake(prob, p = p, tspan = (tsteps[i],tsteps[i+1]), u0 = ode_data[:,i]), Tsit5(),saveat = tsteps[i:i+1]) for i in 1:datasize-1]

		# calculate multiple shooting loss from the single shoots done in above step
		for i in 1:datasize-1
		tot_loss += loss_function(ode_data[:,i:i+1], grp_predictions[i][:, 1:grp_size]) + (continuity_term * sum(abs,grp_predictions[i][:,2] - ode_data[:,i+1]))
		end

		# single shooting predictions from ode_data[:,1] (= u0)
		pred = solve(remake(prob, p = p, tspan = (tsteps[1],tsteps[datasize]), u0 = ode_data[:,1]), Tsit5(),saveat = tsteps)
		return tot_loss, pred, grp_predictions
	end

	# multiple shooting predictions
	grp_predictions = [solve(remake(prob, p = p, tspan = (tsteps[i],tsteps[i+grp_size-1]), u0 = ode_data[:,i]), Tsit5(),saveat = tsteps[i:i+grp_size-1]) for i in 1:grp_size-1:datasize-grp_size]

	# calculate multiple shooting loss
	for i in 1:grp_size-1:datasize-grp_size
		# The term `trunc(Integer,(i-1)/(grp_size-1)+1)` goes from 1, 2, ... , N where N is the total number of groups that can be formed from `ode_data` (In other words, N = trunc(Integer, (datasize-1)/(grp_size-1)))
		tot_loss += loss_function(ode_data[:,i:i+grp_size-1], grp_predictions[trunc(Integer,(i-1)/(grp_size-1)+1)][:, 1:grp_size]) + (continuity_term * sum(abs,grp_predictions[trunc(Integer,(i-1)/(grp_size-1)+1)][:,grp_size] - ode_data[:,i+grp_size-1]))
	end

	# single shooting prediction
	pred = solve(remake(prob, p = p, tspan = (tsteps[1],tsteps[datasize]), u0 = ode_data[:,1]), Tsit5(),saveat = tsteps)
	return tot_loss, pred, grp_predictions
end
