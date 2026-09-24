using DiffEqFlux, BenchmarkTools
using Lux, ComponentArrays, StableRNGs
using OrdinaryDiffEqTsit5

const SUITE = BenchmarkGroup()
const rng = StableRNG(123)

tspan = (0.0f0, 1.0f0)
dudt = Chain(Dense(2 => 8, tanh), Dense(8 => 2))
dudt_aug = Chain(Dense(4 => 8, tanh), Dense(8 => 4))
u0 = Float32[2.0; 0.0]

# =============================================================================
# NeuralODE — construction, setup, forward solve
# =============================================================================

node = NeuralODE(dudt, tspan, Tsit5(); saveat = 0.1f0)
anode = AugmentedNDELayer(NeuralODE(dudt_aug, tspan, Tsit5()), 2)

pd, st = Lux.setup(rng, node)
pda, sta = Lux.setup(rng, anode)
pd = ComponentArray(pd)
pda = ComponentArray(pda)

SUITE["neural_ode"] = BenchmarkGroup()

SUITE["neural_ode"]["construct"] = @benchmarkable NeuralODE(
    $dudt, $tspan, Tsit5(); saveat = 0.1f0
)
SUITE["neural_ode"]["setup"] = @benchmarkable Lux.setup($rng, $node)
SUITE["neural_ode"]["forward"] = @benchmarkable first($node($u0, $pd, $st))
SUITE["neural_ode"]["forward_augmented"] = @benchmarkable first(
    $anode($u0, $pda, $sta)
)

# =============================================================================
# Collocation — non-neural two-stage derivative estimation
# =============================================================================

t = range(0.0f0, 5.0f0; length = 50)
data = sin.(t) .+ 0.01f0 .* randn(rng, Float32, length(t))
data2d = Float32[sin.(t)'; cos.(t)']

SUITE["collocate"] = BenchmarkGroup()

SUITE["collocate"]["kernel"] = @benchmarkable collocate_data(
    $data2d, $t, EpanechnikovKernel()
)
SUITE["collocate"]["triangular"] = @benchmarkable collocate_data(
    $data2d, $t, TriangularKernel()
)
