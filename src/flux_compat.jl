for de_model in (:FFJORD, :HamiltonianNN, :NeuralODE, :NeuralCDDE, :NeuralDAE, :NeuralODEMM)
    @eval begin
        @deprecate $de_model(model, args...; kwargs...) begin
            new_model = Lux.FluxLayer(model)
            ps, st = LuxCore.setup(Random.default_rng(), new_model)
            $de_model(new_model, args...; p=ComponentArray(ps), st=st, kwargs...)
        end

        @functor $de_model (p,)

        function (n::$de_model)(x)
            n(x, n.p, n.st)[1]
        end
    end
end

for de_model2 in (:NeuralDSDE, :NeuralSDE)
    @eval begin
        @deprecate $de_model2(model, model2, args...; kwargs...) begin
            new_model = Lux.FluxLayer(model)
            ps, st = LuxCore.setup(Random.default_rng(), new_model)
            new_model2 = Lux.FluxLayer(model2)
            ps2, st2 = LuxCore.setup(Random.default_rng(), new_model2)
            $de_model2(new_model, new_model2, args...; p=ComponentArray((drift=ps, diffusion=ps2)), st=(drift=st, diffusion=st2), kwargs...)
        end

        @functor $de_model2 (p,)

        function (n::$de_model2)(x)
            n(x, n.p, n.st)[1]
        end
    end
end
