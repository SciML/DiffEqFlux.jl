for de_model in (:FFJORD, :HamiltonianNN, :NeuralODE, :NeuralCDDE, :NeuralDAE, :NeuralODEMM)
    @eval begin
        @deprecate $de_model(model, args...; kwargs...) begin
            new_model = Lux.transform(model)
            ps, st = LuxCore.setup(Random.default_rng(), new_model)
            $de_model(new_model, args...; p=ps, kwargs...)
        end
        @functor $de_model (p,)
    end
end

for de_model2 in (:NeuralDSDE, :NeuralSDE)
    @eval begin
        @deprecate $de_model2(model, model2, args...; kwargs...) begin
            new_model = Lux.transform(model)
            ps, st = LuxCore.setup(Random.default_rng(), new_model)
            new_model2 = Lux.transform(model2)
            ps2, st2 = LuxCore.setup(Random.default_rng(), new_model2)
            $de_model2(new_model, new_model2, args...; p=[ps; ps2], kwargs...)
        end
        @functor $de_model2 (p,)
    end
end
