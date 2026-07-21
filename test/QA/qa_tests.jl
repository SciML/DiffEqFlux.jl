using SciMLTesting, DiffEqFlux

# DiffEqFlux historically exposes these dependency-owned names. Keep the snapshot
# explicit so dependency upgrades cannot silently add more public reexports.
const REEXPORTED_API = (
    Symbol("@compact"), Symbol("@init_fn"), Symbol("@non_trainable"), :ADTypes,
    :AMDGPUDevice, :API, :AbstractADType, :AbstractColoringAlgorithm,
    :AbstractLuxContainerLayer, :AbstractLuxLayer, :AbstractLuxWrapperLayer, :AbstractSparsityDetector,
    :AdaptiveLPPool, :AdaptiveMaxPool, :AdaptiveMeanPool, :AdjointLSS,
    :AlphaDropout, :AlternatePrecision, :AutoChainRules, :AutoDiffractor,
    :AutoEnzyme, :AutoFastDifferentiation, :AutoFiniteDiff, :AutoFiniteDifferences,
    :AutoForwardDiff, :AutoGTPSA, :AutoHyperHessians, :AutoModelingToolkit,
    :AutoMooncake, :AutoMooncakeForward, :AutoPolyesterForwardDiff, :AutoReactant,
    :AutoReverseDiff, :AutoSparse, :AutoSparseFastDifferentiation, :AutoSparseFiniteDiff,
    :AutoSparseForwardDiff, :AutoSparsePolyesterForwardDiff, :AutoSparseReverseDiff, :AutoSparseZygote,
    :AutoSymbolics, :AutoTapir, :AutoTaylorDiff, :AutoTracker,
    :AutoZygote, :BacksolveAdjoint, :Basis, :BatchLastIndex,
    :BatchNorm, :BidirectionalRNN, :Bilinear, :BinaryCrossEntropyLoss,
    :BinaryFocalLoss, :Boltz, :BranchLayer, :CPUDevice,
    :CUDADevice, :Chain, :CompactLuxLayer, :Conv,
    :ConvDims, :ConvTranspose, :CrossEntropyLoss, :Dense,
    :DenseConvDims, :DeviceIterator, :DiceCoeffLoss, :DistributedUtils,
    :Dropout, :Embedding, :EnzymeVJP, :FlattenLayer,
    :FluxLayer, :FocalLoss, :ForwardDiffOverAdjoint, :ForwardDiffSensitivity,
    :ForwardLSS, :ForwardSensitivity, :FromFluxAdaptor, :GRUCell,
    :GaussAdjoint, :GenericLossFunction, :GlobalLPPool, :GlobalMaxPool,
    :GlobalMeanPool, :GroupNorm, :HingeLoss, :HuberLoss,
    :InstanceNorm, :InterpolatingAdjoint, :KLDivergenceLoss, :L1Loss,
    :L2Loss, :LPPool, :LSTMCell, :LayerNorm,
    :Layers, :Lux, :LuxCore, :LuxLib,
    :LuxOps, :MAELoss, :MLDataDevices, :MPIBackend,
    :MSELoss, :MSLELoss, :MaxPool, :Maxout,
    :MeanPool, :MetalDevice, :MultiHeadAttention, :NCCLBackend,
    :NILSAS, :NILSS, :NoAutoDiff, :NoAutoDiffSelectedError,
    :NoOpLayer, :OpenCLDevice, :PIML, :PairwiseFusion,
    :Parallel, :PixelShuffle, :PoissonLoss, :PoolDims,
    :QuadratureAdjoint, :RMSNorm, :RNNCell, :ReactantDevice,
    :Recurrence, :RepeatedLayer, :ReshapeLayer, :ReverseDiffAdjoint,
    :ReverseDiffVJP, :ReverseSequence, :RotaryPositionalEmbedding, :SamePad,
    :Scale, :SelectDim, :SiameseContrastiveLoss, :SimpleChainsLayer,
    :SinusoidalPositionalEmbedding, :SkipConnection, :SquaredHingeLoss, :StatefulLuxLayer,
    :StatefulRecurrentCell, :SteadyStateAdjoint, :TimeLastIndex, :ToSimpleChainsAdaptor,
    :TrackerAdjoint, :TrackerVJP, :Training, :Upsample,
    :VariationalHiddenDropout, :Vision, :WeightInitializers, :WeightNorm,
    :WrappedFunction, :XLADevice, :ZygoteAdjoint, :ZygoteVJP,
    :alpha_dropout, :apply_rotary_embedding, :batched_adjoint, :batched_jacobian,
    :batched_matmul, :batched_mul, :batched_mul!, :batched_transpose,
    :batched_vec, :batchnorm, :bf16, :bias_act!,
    :bias_activation, :bias_activation!!, :celu, :column_coloring,
    :compute_rotary_embedding_params, :conv, :conv!, :conv_bias_act,
    :conv_bias_act!, :cpu_device, :default_device_rng, :dot_product_attention,
    :dot_product_attention_scores, :dropout, :elu, :f16,
    :f32, :f64, :fast_activation, :fast_activation!!,
    :fused_conv_bias_activation, :fused_dense_bias_activation, :gelu, :get_device,
    :get_device_type, :glorot_normal, :glorot_uniform, :glu,
    :gpu_backend!, :gpu_device, :groupnorm, :hardsigmoid,
    :hardswish, :hardtanh, :hardσ, :hessian_sparsity,
    :identity_init, :imrotate, :instancenorm, :jacobian_sparsity,
    :jacobian_vector_product, :kaiming_normal, :kaiming_uniform, :layernorm,
    :leakyrelu, :lisht, :logcosh, :logsigmoid,
    :logsoftmax, :logsoftmax!, :logsumexp, :logσ,
    :lpnormpool, :lpnormpool!, :make_causal_mask, :match_eltype,
    :maxpool, :maxpool!, :meanpool, :meanpool!,
    :mish, :oneAPIDevice, :ones16, :ones32,
    :ones64, :onesC16, :onesC32, :onesC64,
    :orthogonal, :pad_circular, :pad_constant, :pad_reflect,
    :pad_repeat, :pad_symmetric, :pad_zeros, :pixel_shuffle,
    :rand16, :rand32, :rand64, :randC16,
    :randC32, :randC64, :randn16, :randn32,
    :randn64, :randnC16, :randnC32, :randnC64,
    :reactant_device, :recursive_add!!, :recursive_copyto!, :recursive_make_zero,
    :recursive_make_zero!!, :recursive_map, :relu, :relu6,
    :reset_gpu_device!, :row_coloring, :rrelu, :scaled_dot_product_attention,
    :selu, :sigmoid, :sigmoid_fast, :softmax,
    :softmax!, :softplus, :softshrink, :softsign,
    :sparse_init, :supported_gpu_backends, :swish, :symmetric_coloring,
    :tanh_fast, :tanhshrink, :thresholdrelu, :transform,
    :trelu, :truncated_normal, :upsample_bilinear, :upsample_linear,
    :upsample_nearest, :upsample_trilinear, :vector_jacobian_product, :xla_device,
    :zeros16, :zeros32, :zeros64, :zerosC16,
    :zerosC32, :zerosC64, :σ, :∇conv_data,
    :∇conv_data!, :∇conv_filter, :∇conv_filter!, :∇imrotate,
    :∇logsoftmax, :∇logsoftmax!, :∇lpnormpool, :∇lpnormpool!,
    :∇maxpool, :∇maxpool!, :∇meanpool, :∇meanpool!,
    :∇softmax, :∇softmax!, :∇upsample_bilinear, :∇upsample_linear,
    :∇upsample_nearest, :∇upsample_trilinear,
)

const UNDOCUMENTED_REEXPORTS = (
    :AMDGPUDevice, :AutoModelingToolkit, :AutoSparseFastDifferentiation, :AutoSparseFiniteDiff,
    :AutoSparseForwardDiff, :AutoSparsePolyesterForwardDiff, :AutoSparseReverseDiff, :AutoSparseZygote,
    :BatchLastIndex, :CPUDevice, :CUDADevice, :CompactLuxLayer,
    :MetalDevice, :OpenCLDevice, :ReactantDevice, :SamePad,
    :TimeLastIndex, :XLADevice, :conv!, :conv_bias_act,
    :conv_bias_act!, :logsoftmax!, :lpnormpool!, :maxpool!,
    :meanpool!, :softmax!, :∇logsoftmax, :∇logsoftmax!,
    :oneAPIDevice, :transform, :xla_device, :∇conv_data,
    :∇conv_data!, :∇conv_filter, :∇conv_filter!, :∇lpnormpool,
    :∇lpnormpool!, :∇maxpool, :∇maxpool!, :∇meanpool,
    :∇meanpool!, :∇softmax, :∇softmax!,
)

const REEXPORTS_WITH_INHERITED_RENDERING = (
    :ADTypes, :API, :Basis, :Boltz,
    :DistributedUtils, :Layers, :Lux, :LuxCore,
    :LuxLib, :LuxOps, :MLDataDevices, :PIML,
    :Training, :Vision, :WeightInitializers,
)

const UNRENDERED_REEXPORTS = Tuple(
    name for name in REEXPORTED_API if !(name in REEXPORTS_WITH_INHERITED_RENDERING)
)

run_qa(
    DiffEqFlux;
    reexports_allow = REEXPORTED_API,
    # `ambiguities = false` in test_all + a separate non-recursive ambiguity check
    # historically; keep ambiguities on but non-recursive (recursive hits the deep
    # Lux/SciMLSensitivity stack and is not DiffEqFlux's responsibility).
    aqua_kwargs = (; ambiguities = (; recursive = false)),
    api_docs_kwargs = (;
        ignore = UNDOCUMENTED_REEXPORTS,
        rendered_ignore = UNRENDERED_REEXPORTS,
    ),
    ei_kwargs = (
        # `FFJORDDistribution` implements Distributions' documented extension points
        # `_logpdf`/`_rand!` (a custom `ContinuousMultivariateDistribution` must define
        # these; see `Distributions.common`: "Instead of `logpdf` one should implement
        # `_logpdf(d, x)`"). They are deliberately underscore-prefixed and not public,
        # so the access can be neither migrated to a public owner nor made public.
        all_qualified_accesses_are_public = (;
            ignore = (
                :_logpdf,  # Distributions extension point (non-public by convention)
                :_rand!,   # Distributions extension point (non-public by convention)
            ),
        ),
    ),
)
