"""
Returns the predictions on the whole ODE data and a total loss after trying a 'Direct multiple shooting' on ODE data.
In Direct Multiple Shooting, the Neural Network divides the interval into smaller intervals and solves for them separately.
The default group size is 5 implying the whole dataset would be divided in groups of 5 and the Neural Network will solve on them individually.
The default continuity term is 100 implying any losses arising from the non-continuity of 2 different groups will be scaled by 100.

```julia
loss_neuralode_helper(p,ode_data,tsteps,prob,loss_function,grp_size=5,continuity_term=100)
```
Arguments:
- `p`: The parameters of the Neural Network to be trained.
- `ode_data`: Original Data to be modelled.
- `tsteps`: Timesteps on which ode_data was calculated.
- `prob`: ODE problem that the Neural Network attempts to solve.
- `loss_function`: Any arbitrary function to calculate loss.
- `grp_size`: The group size achieved after splitting the ode_data into equal sizes.
- `continuity_term`: Multiplying factor to ensure continuity of predictions throughout different groups.

!!!note
		The parameter 'continuity_term' should be a relatively big number to enforce a large penalty whenever the last point of any group doesn't coincide with the first point of next group.
"""
function loss_neuralode_helper(p::Array,ode_data::Array,tsteps::Array,prob::ODEProblem,loss_function::Function,grp_size::Integer=5,continuity_term::Integer=100)
	tot_loss = 0
	datasize = length(ode_data[1,:])
	if(grp_size == 1)
		for i in 1:datasize-1
			pred = Array(solve(remake(prob, p=p, tspan=(tsteps[i],tsteps[i+1]), u0=ode_data[:,i]), Tsit5(),saveat = tsteps[i:i+1]))
			tot_loss += loss_function(ode_data[:,i:i+1], pred[:,1:2]) + (continuity_term * sum(abs,ode_data[:,i+ 1]-pred[:,2]))
		end

		pred = solve(remake(prob, p = p, tspan = (tsteps[1],tsteps[datasize]), u0 = ode_data[:,1]), Tsit5(),saveat = tsteps)
		return tot_loss, pred
	end

	for i in 1:grp_size-1:datasize-grp_size
		pred = solve(remake(prob, p = p, tspan = (tsteps[i],tsteps[i+grp_size-1]), u0 = ode_data[:,i]), Tsit5(),saveat = tsteps[i:i+grp_size-1])
		tot_loss += loss_function(ode_data[:,i:i+grp_size-1], pred[:,1:grp_size]) + (continuity_term * sum(abs,pred[:,grp_size] - ode_data[:,i+grp_size-1]))
	end
	pred = solve(remake(prob, p = p, tspan = (tsteps[1],tsteps[datasize]), u0 = ode_data[:,1]), Tsit5(),saveat = tsteps)
	return tot_loss, pred
end
