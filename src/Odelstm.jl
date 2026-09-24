using Lux
using ComponentArrays
using Optimisers
using LinearAlgebra
using Random
using Statistics
using MLDatasets
using DelimitedFiles
using Printf

export ODELSTMCell, ODELSTMModel, IrregularSequenceLearner
export PersonData, ETSMnistData, XORData
export train_model!, evaluate_model, load_dataset

mutable struct ODELSTMCell{F,S}
    lstm_cell::Lux.LSTMCell
    f_node::F
    solver::S
    input_size::Int
    hidden_size::Int
end

function ODELSTMCell(input_size::Int, hidden_size::Int, solver)
    lstm_cell = Lux.LSTMCell(input_size, hidden_size)
    f_node = Chain(
        Dense(hidden_size, hidden_size, tanh),
        Dense(hidden_size, hidden_size)
    )
    return ODELSTMCell(lstm_cell, f_node, solver, input_size, hidden_size)
end

function (cell::ODELSTMCell)(x, state, ts, p, st)
    h, c = state
    new_h, new_c, st = cell.lstm_cell(x, (h, c), p, st)
    
    if !(cell.solver isa Union{Tsit5,DP5,BS3})
        evolved_h, st = solve_fixed_step(cell, new_h, ts, p, st)
    else
        evolved_h, st = solve_adaptive(cell, new_h, ts, p, st)
    end
    
    return evolved_h, (evolved_h, new_c), st
end

function solve_fixed_step(cell::ODELSTMCell, h, ts, p, st)
    prob = ODEProblem((u,p,t)->cell.f_node(u,p,st)[1], h, (0.0f0, Float32(ts)))
    sol = solve(prob, cell.solver; adaptive=false)
    return sol.u[end], st
end

function solve_adaptive(cell::ODELSTMCell, h, ts, p, st)
    if ndims(h) == 2
        batch_size = size(h, 2)
        results = similar(h)
        
        for i in 1:batch_size
            h_i = h[:, i]
            ts_i = ts isa AbstractVector ? ts[i] : ts
            t_span = (0.0f0, Float32(ts_i))
            
            prob = ODEProblem((u,p,t)->cell.f_node(u,p,st)[1], h_i, t_span)
            sol = solve(prob, cell.solver, saveat=[t_span[2]], dense=false)
            results[:, i] = sol.u[end]
        end
        return results, st
    else
        t_span = (0.0f0, Float32(ts))
        prob = ODEProblem((u,p,t)->cell.f_node(u,p,st)[1], h, t_span)
        sol = solve(prob, cell.solver, saveat=[t_span[2]], dense=false)
        return sol.u[end], st
    end
end

struct ODELSTMModel{C,O}
    rnn_cell::C
    output_layer::O
    return_sequences::Bool
end

function ODELSTMModel(in_features::Int, hidden_size::Int, out_features::Int; 
                      return_sequences::Bool=true, solver)
    rnn_cell = ODELSTMCell(in_features, hidden_size, solver)
    output_layer = Dense(hidden_size, out_features)
    return ODELSTMModel(rnn_cell, output_layer, return_sequences)
end

mutable struct IrregularSequenceLearner{M,O,S}
    model::M
    optimizer::O
    states::S
    train_losses::Vector{Float32}
    val_losses::Vector{Float32}
    train_accs::Vector{Float32}
    val_accs::Vector{Float32}
end

function IrregularSequenceLearner(model, lr=0.005f0; rng=Random.default_rng())
    optimizer = Optimisers.Adam(lr)
    ps, st = Lux.setup(rng, model)
    return IrregularSequenceLearner(
        model, optimizer, st,
        Float32[], Float32[], Float32[], Float32[]
    )
end

function train_step!(learner::IrregularSequenceLearner, batch, p)
    if length(batch) == 4
        x, t, y, mask = batch
    else
        x, t, y = batch
        mask = nothing
    end
    
    (loss, (y_hat, st)), grads = value_and_gradient(p) do p
        y_hat, st = learner.model(x, t, mask, p, learner.states)
        Lux.crossentropy(y_hat, Flux.onehotbatch(reshape(y, :), 0:maximum(y)))
    end
    
    learner.states = st
    p = Optimisers.update(learner.optimizer, p, grads)
    
    preds = Flux.onecold(y_hat) .- 1
    acc = mean(preds .== reshape(y, :))
    
    return loss, acc, p
end

function validation_step(learner::IrregularSequenceLearner, batch, p)
    if length(batch) == 4
        x, t, y, mask = batch
    else
        x, t, y = batch
        mask = nothing
    end
    
    y_hat, st = learner.model(x, t, mask, p, learner.states)
    
    loss = Lux.crossentropy(y_hat, Flux.onehotbatch(reshape(y, :), 0:maximum(y)))
    preds = Flux.onecold(y_hat) .- 1
    acc = mean(preds .== reshape(y, :))
    
    return loss, acc
end

function train_model!(learner::IrregularSequenceLearner, train_loader, val_loader=nothing; epochs=100)
    p = ComponentArray(Lux.initialparameters(Random.default_rng(), learner.model))
    for epoch in 1:epochs
        train_loss_epoch = 0.0f0
        train_acc_epoch = 0.0f0
        train_batches = 0
        
        for batch in train_loader
            loss, acc, p = train_step!(learner, batch, p)
            train_loss_epoch += loss
            train_acc_epoch += acc
            train_batches += 1
        end
        
        avg_train_loss = train_loss_epoch / train_batches
        avg_train_acc = train_acc_epoch / train_batches
        
        push!(learner.train_losses, avg_train_loss)
        push!(learner.train_accs, avg_train_acc)
        
        if val_loader !== nothing
            val_loss_epoch = 0.0f0
            val_acc_epoch = 0.0f0
            val_batches = 0
            
            for batch in val_loader
                loss, acc = validation_step(learner, batch, p)
                val_loss_epoch += loss
                val_acc_epoch += acc
                val_batches += 1
            end
            
            avg_val_loss = val_loss_epoch / val_batches
            avg_val_acc = val_acc_epoch / val_batches
            
            push!(learner.val_losses, avg_val_loss)
            push!(learner.val_accs, avg_val_acc)
            
            if epoch % 10 == 0
                @printf("Epoch %d: Train Loss=%.4f, Train Acc=%.4f, Val Loss=%.4f, Val Acc=%.4f\n", 
                        epoch, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc)
            end
        else
            if epoch % 10 == 0
                @printf("Epoch %d: Train Loss=%.4f, Train Acc=%.4f\n", 
                        epoch, avg_train_loss, avg_train_acc)
            end
        end
    end
end

struct PersonData
    train_x::Array{Float32,3}
    train_y::Array{Int32,2}
    train_t::Array{Float32,3}
    test_x::Array{Float32,3}
    test_y::Array{Int32,2}
    test_t::Array{Float32,3}
    feature_size::Int
    num_classes::Int
end

function PersonData(; seq_len::Int=32)
    n_train, n_test = 1000, 200
    feature_size = 7
    num_classes = 7
    
    train_x = randn(Float32, n_train, seq_len, feature_size)
    train_y = rand(Int32(0):Int32(num_classes-1), n_train, seq_len)
    train_t = rand(Float32, n_train, seq_len, 1) .* 2.0f0
    
    test_x = randn(Float32, n_test, seq_len, feature_size)
    test_y = rand(Int32(0):Int32(num_classes-1), n_test, seq_len)
    test_t = rand(Float32, n_test, seq_len, 1) .* 2.0f0
    
    return PersonData(train_x, train_y, train_t, test_x, test_y, test_t, feature_size, num_classes)
end

struct ETSMnistData
    train_events::Array{Float32,3}
    train_elapsed::Array{Float32,3}
    train_mask::Array{Bool,2}
    train_y::Vector{Int32}
    test_events::Array{Float32,3}
    test_elapsed::Array{Float32,3}
    test_mask::Array{Bool,2}
    test_y::Vector{Int32}
    pad_size::Int
end

function ETSMnistData(; pad_size::Int=256)
    train_x, train_y = MLDatasets.MNIST(:train)[:]
    test_x, test_y = MLDatasets.MNIST(:test)[:]
    threshold = 128
    
    function transform_sample(x)
        x_flat = vec(x)
        events = zeros(Float32, pad_size, 1)
        elapsed = zeros(Float32, pad_size, 1)
        mask = zeros(Bool, pad_size)
        
        last_char = -1
        write_index = 1
        elapsed_counter = 0
        
        for i in 1:length(x_flat)
            elapsed_counter += 1
            char = Int(x_flat[i] > threshold)
            
            if last_char != char
                if write_index <= pad_size
                    events[write_index, 1] = Float32(char)
                    elapsed[write_index, 1] = Float32(elapsed_counter)
                    mask[write_index] = true
                    write_index += 1
                    elapsed_counter = 0
                end
                if write_index > pad_size
                    break
                end
            end
            last_char = char
        end
        
        return events, elapsed, mask
    end
    
    train_events_list = []
    train_elapsed_list = []
    train_mask_list = []
    
    for i in 1:size(train_x, 3)
        events, elapsed, mask = transform_sample(train_x[:, :, i])
        push!(train_events_list, events)
        push!(train_elapsed_list, elapsed)
        push!(train_mask_list, mask)
    end
    
    test_events_list = []
    test_elapsed_list = []
    test_mask_list = []
    
    for i in 1:size(test_x, 3)
        events, elapsed, mask = transform_sample(test_x[:, :, i])
        push!(test_events_list, events)
        push!(test_elapsed_list, elapsed)
        push!(test_mask_list, mask)
    end
    
    train_events = cat(train_events_list..., dims=3)
    train_elapsed = cat(train_elapsed_list..., dims=3)
    train_mask = hcat(train_mask_list...)
    
    test_events = cat(test_events_list..., dims=3)
    test_elapsed = cat(test_elapsed_list..., dims=3)
    test_mask = hcat(test_mask_list...)
    
    train_elapsed ./= pad_size
    test_elapsed ./= pad_size
    
    train_events = permutedims(train_events, (3, 1, 2))
    train_elapsed = permutedims(train_elapsed, (3, 1, 2))
    train_mask = train_mask'
    
    test_events = permutedims(test_events, (3, 1, 2))
    test_elapsed = permutedims(test_elapsed, (3, 1, 2))
    test_mask = test_mask'
    
    return ETSMnistData(
        train_events, train_elapsed, train_mask, Int32.(train_y),
        test_events, test_elapsed, test_mask, Int32.(test_y),
        pad_size
    )
end

struct XORData
    train_events::Array{Float32,3}
    train_elapsed::Array{Float32,3}
    train_mask::Array{Bool,2}
    train_y::Vector{Int32}
    test_events::Array{Float32,3}
    test_elapsed::Array{Float32,3}
    test_mask::Array{Bool,2}
    test_y::Vector{Int32}
    pad_size::Int
end

function XORData(; pad_size::Int=24, event_based::Bool=true)
    function create_sample(rng, event_based)
        events = zeros(Float32, pad_size, 1)
        elapsed = zeros(Float32, pad_size, 1)
        mask = zeros(Bool, pad_size)
        
        length_seq = rand(rng, 2:pad_size)
        label = 0
        write_index = 1
        elapsed_counter = 0
        last_char = -1
        
        for i in 1:length_seq
            elapsed_counter += 1
            char = rand(rng, 0:1)
            label += char
            
            if event_based
                if last_char != char && write_index <= pad_size
                    events[write_index, 1] = Float32(char)
                    elapsed[write_index, 1] = Float32(elapsed_counter)
                    mask[write_index] = true
                    write_index += 1
                    elapsed_counter = 0
                end
            else
                if write_index <= pad_size
                    events[write_index, 1] = Float32(char)
                    elapsed[write_index, 1] = Float32(elapsed_counter)
                    mask[write_index] = true
                    write_index += 1
                    elapsed_counter = 0
                end
            end
            last_char = char
        end
        
        if event_based && elapsed_counter > 0 && write_index <= pad_size
            events[write_index, 1] = Float32(last_char)
            elapsed[write_index, 1] = Float32(elapsed_counter)
            mask[write_index] = true
        end
        
        label = label % 2
        return events, elapsed, mask, Int32(label)
    end
    
    rng_train = MersenneTwister(1234984)
    train_size = 100000
    train_events_list = []
    train_elapsed_list = []
    train_mask_list = []
    train_y_list = []
    
    for i in 1:train_size
        events, elapsed, mask, label = create_sample(rng_train, event_based)
        push!(train_events_list, events)
        push!(train_elapsed_list, elapsed)
        push!(train_mask_list, mask)
        push!(train_y_list, label)
    end
    
    rng_test = MersenneTwister(48736)
    test_size = 10000
    test_events_list = []
    test_elapsed_list = []
    test_mask_list = []
    test_y_list = []
    
    for i in 1:test_size
        events, elapsed, mask, label = create_sample(rng_test, event_based)
        push!(test_events_list, events)
        push!(test_elapsed_list, elapsed)
        push!(test_mask_list, mask)
        push!(test_y_list, label)
    end
    
    train_events = cat(train_events_list..., dims=3)
    train_elapsed = cat(train_elapsed_list..., dims=3)
    train_mask = hcat(train_mask_list...)
    
    test_events = cat(test_events_list..., dims=3)
    test_elapsed = cat(test_elapsed_list..., dims=3)
    test_mask = hcat(test_mask_list...)
    
    train_elapsed ./= pad_size
    test_elapsed ./= pad_size
    
    train_events = permutedims(train_events, (3, 1, 2))
    train_elapsed = permutedims(train_elapsed, (3, 1, 2))
    train_mask = train_mask'
    
    test_events = permutedims(test_events, (3, 1, 2))
    test_elapsed = permutedims(test_elapsed, (3, 1, 2))
    test_mask = test_mask'
    
    return XORData(
        train_events, train_elapsed, train_mask, train_y_list,
        test_events, test_elapsed, test_mask, test_y_list,
        pad_size
    )
end

function load_dataset(dataset_name::String; kwargs...)
    if dataset_name == "person"
        dataset = PersonData(; kwargs...)
        return_sequences = true
        
        train_data = [(dataset.train_x[i:i, :, :], dataset.train_t[i:i, :, :], dataset.train_y[i:i, :]) 
                      for i in 1:size(dataset.train_x, 1)]
        test_data = [(dataset.test_x[i:i, :, :], dataset.test_t[i:i, :, :], dataset.test_y[i:i, :]) 
                     for i in 1:size(dataset.test_x, 1)]
        
        in_features = dataset.feature_size
        num_classes = dataset.num_classes
        
    elseif dataset_name == "et_mnist"
        dataset = ETSMnistData(; kwargs...)
        return_sequences = false
        
        train_data = [(dataset.train_events[i:i, :, :], dataset.train_elapsed[i:i, :, :], 
                       [dataset.train_y[i]], dataset.train_mask[i:i, :]) 
                      for i in 1:size(dataset.train_events, 1)]
        test_data = [(dataset.test_events[i:i, :, :], dataset.test_elapsed[i:i, :, :], 
                      [dataset.test_y[i]], dataset.test_mask[i:i, :]) 
                     for i in 1:size(dataset.test_events, 1)]
        
        in_features = 1
        num_classes = 10
        
    elseif dataset_name == "xor"
        dataset = XORData(; kwargs...)
        return_sequences = false
        
        train_data = [(dataset.train_events[i:i, :, :], dataset.train_elapsed[i:i, :, :], 
                       [dataset.train_y[i]], dataset.train_mask[i:i, :]) 
                      for i in 1:length(dataset.train_y)]
        test_data = [(dataset.test_events[i:i, :, :], dataset.test_elapsed[i:i, :, :], 
                      [dataset.test_y[i]], dataset.test_mask[i:i, :]) 
                     for i in 1:length(dataset.test_y)]
        
        in_features = 1
        num_classes = 2
        
    else
        throw(ArgumentError("Unknown dataset: $dataset_name"))
    end
    
    return train_data, test_data, in_features, num_classes, return_sequences
end

function main_training_loop(; dataset="person", solver=Tsit5(), size=64, epochs=100, lr=0.01f0)
    train_loader, test_loader, in_features, num_classes, return_sequences = load_dataset(dataset)
    
    model = ODELSTMModel(in_features, size, num_classes; 
                         return_sequences=return_sequences, solver=solver)
    
    learner = IrregularSequenceLearner(model, lr)
    
    train_model!(learner, train_loader; epochs=epochs)
    
    results = evaluate_model(learner, test_loader)
    
    return learner, results
end