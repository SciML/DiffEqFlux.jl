# Hamiltonian Neural Network

```julia
using DiffEqFlux, Flux, OrdinaryDiffEq

t = range(0.0, 1.0, length = 100)
q_t = reshape(sin.(2π * t), 1, :)
p_t = reshape(cos.(2π * t), 1, :)

data = cat(q_t, p_t, dims = 1)
u0 = data[:, 1:1]

model = NeuralHamiltonianDE(
    Chain(Dense(2, 100, relu), Dense(100, 1)),
    (0.0f0, 1.0f0),
    Tsit5(), save_everystep = false,
    save_start = true,
    saveat = t
)

diffeqarray_to_array(x) = Array(x)[:, 1, :]

predict(x) = diffeqarray_to_array(model(x))

loss() = sum((data .- predict(u0)) .^ 2)

callback() = println("Loss Neural Hamiltonian DE = $(loss())")

dummy_data = Iterators.repeated((), 1000)
Flux.train!(loss, Flux.params(model), dummy_data, ADAM(0.001), cb = callback)
```

## Expected Output

```julia
```