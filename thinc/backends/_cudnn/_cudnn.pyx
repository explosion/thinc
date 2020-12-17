###############################################################################
# Extern
###############################################################################
from libc.stdint cimport int32_t, uint32_t
from . cimport driver

cpdef enum:
    CUDNN_DATA_FLOAT = 0
    CUDNN_DATA_DOUBLE = 1
    CUDNN_DATA_HALF = 2

    CUDNN_DEFAULT_MATH = 0
    CUDNN_TENSOR_OP_MATH = 1

    CUDNN_NOT_PROPAGATE_NAN = 0
    CUDNN_PROPAGATE_NAN = 1

    CUDNN_NON_DETERMINISTIC = 0
    CUDNN_DETERMINISTIC = 1

    CUDNN_TENSOR_NCHW = 0
    CUDNN_TENSOR_NHWC = 1

    CUDNN_OP_TENSOR_ADD = 0
    CUDNN_OP_TENSOR_MUL = 1
    CUDNN_OP_TENSOR_MIN = 2
    CUDNN_OP_TENSOR_MAX = 3
    CUDNN_OP_TENSOR_SQRT = 4
    CUDNN_OP_TENSOR_NOT = 5

    CUDNN_REDUCE_TENSOR_ADD = 0
    CUDNN_REDUCE_TENSOR_MUL = 1
    CUDNN_REDUCE_TENSOR_MIN = 2
    CUDNN_REDUCE_TENSOR_MAX = 3
    CUDNN_REDUCE_TENSOR_AMAX = 4
    CUDNN_REDUCE_TENSOR_AVG = 5
    CUDNN_REDUCE_TENSOR_NORM1 = 6
    CUDNN_REDUCE_TENSOR_NORM2 = 7
    CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS = 8

    CUDNN_REDUCE_TENSOR_NO_INDICES = 0
    CUDNN_REDUCE_TENSOR_FLATTENED_INDICES = 1

    CUDNN_32BIT_INDICES = 0
    CUDNN_64BIT_INDICES = 1
    CUDNN_16BIT_INDICES = 2
    CUDNN_8BIT_INDICES = 3

    CUDNN_ADD_IMAGE = 0
    CUDNN_ADD_SAME_HW = 0
    CUDNN_ADD_FEATURE_MAP = 1
    CUDNN_ADD_SAME_CHW = 1
    CUDNN_ADD_SAME_C = 2
    CUDNN_ADD_FULL_TENSOR = 3

    CUDNN_CONVOLUTION = 0
    CUDNN_CROSS_CORRELATION = 1

    CUDNN_CONVOLUTION_FWD_NO_WORKSPACE = 0
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST = 1
    CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = 2

    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM = 2
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = 3
    CUDNN_CONVOLUTION_FWD_ALGO_FFT = 4
    CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING = 5
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD = 6
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED = 7

    CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE = 0
    CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST = 1
    CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT = 2

    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = 0
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 = 1
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT = 2
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 = 3
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD = 4
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED = 5

    CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE = 0
    CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST = 1
    CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT = 2

    CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = 0
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = 1
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT = 2
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = 3
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD = 4
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = 5

    CUDNN_SOFTMAX_FAST = 0
    CUDNN_SOFTMAX_ACCURATE = 1
    CUDNN_SOFTMAX_LOG = 2

    CUDNN_SOFTMAX_MODE_INSTANCE = 0
    CUDNN_SOFTMAX_MODE_CHANNEL = 1

    CUDNN_POOLING_MAX = 0
    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1
    CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2
    CUDNN_POOLING_MAX_DETERMINISTIC = 3

    CUDNN_ACTIVATION_SIGMOID = 0
    CUDNN_ACTIVATION_RELU = 1
    CUDNN_ACTIVATION_TANH = 2
    CUDNN_ACTIVATION_CLIPPED_RELU = 3
    CUDNN_ACTIVATION_ELU = 4
    CUDNN_ACTIVATION_IDENTITY = 5

    CUDNN_LRN_CROSS_CHANNEL_DIM1 = 0

    CUDNN_DIVNORM_PRECOMPUTED_MEANS = 0

    CUDNN_BATCHNORM_PER_ACTIVATION = 0
    CUDNN_BATCHNORM_SPATIAL = 1
    CUDNN_BATCHNORM_SPATIAL_PERSISTENT = 2

    CUDNN_CTC_LOSS_ALGO_DETERMINISTIC = 0
    CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC = 1

    CUDNN_BATCHNORM_OPS_BN = 0
    CUDNN_BATCHNORM_OPS_BN_ACTIVATION = 1
    CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION = 2

    CUDNN_RNN_RELU = 0
    CUDNN_RNN_TANH = 1
    CUDNN_LSTM = 2
    CUDNN_GRU = 3

    CUDNN_UNIDIRECTIONAL = 0
    CUDNN_BIDIRECTIONAL = 1

    CUDNN_RNN_ALGO_STANDARD = 0
    CUDNN_RNN_ALGO_PERSIST_STATIC = 1
    CUDNN_RNN_ALGO_PERSIST_DYNAMIC = 2

    CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED = 0
    CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED = 1
    CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED = 2

    CUDNN_RNN_PADDED_IO_DISABLED = 0
    CUDNN_RNN_PADDED_IO_ENABLED = 1

    CUDNN_LINEAR_INPUT = 0
    CUDNN_SKIP_INPUT = 1

    CUDNN_SAMPLER_BILINEAR = 0

    CUDNN_STATUS_SUCCESS = 0
    CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING = 11
    CUDNN_STATUS_RUNTIME_IN_PROGRESS = 12
    CUDNN_STATUS_RUNTIME_FP_OVERFLOW = 13

    CUDNN_ERRQUERY_RAWCODE = 0
    CUDNN_ERRQUERY_NONBLOCKING = 1
    CUDNN_ERRQUERY_BLOCKING = 2

    # cudnnFusedOps_t
    CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS = 0
    CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD = 1
    CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING = 2
    CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE = 3
    CUDNN_FUSED_CONV_SCALE_BIAS_ADD_ACTIVATION = 4
    CUDNN_FUSED_SCALE_BIAS_ADD_ACTIVATION_GEN_BITMASK = 5
    CUDNN_FUSED_DACTIVATION_FORK_DBATCHNORM = 6

    # cudnnFusedOpsConstParamLabel_t
    CUDNN_PARAM_XDESC = 0
    CUDNN_PARAM_XDATA_PLACEHOLDER = 1
    CUDNN_PARAM_BN_MODE = 2
    CUDNN_PARAM_BN_EQSCALEBIAS_DESC = 3
    CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER = 4
    CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER = 5
    CUDNN_PARAM_ACTIVATION_DESC = 6
    CUDNN_PARAM_CONV_DESC = 7
    CUDNN_PARAM_WDESC = 8
    CUDNN_PARAM_WDATA_PLACEHOLDER = 9
    CUDNN_PARAM_DWDESC = 10
    CUDNN_PARAM_DWDATA_PLACEHOLDER = 11
    CUDNN_PARAM_YDESC = 12
    CUDNN_PARAM_YDATA_PLACEHOLDER = 13
    CUDNN_PARAM_DYDESC = 14
    CUDNN_PARAM_DYDATA_PLACEHOLDER = 15
    CUDNN_PARAM_YSTATS_DESC = 16
    CUDNN_PARAM_YSUM_PLACEHOLDER = 17
    CUDNN_PARAM_YSQSUM_PLACEHOLDER = 18
    CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC = 19
    CUDNN_PARAM_BN_SCALE_PLACEHOLDER = 20
    CUDNN_PARAM_BN_BIAS_PLACEHOLDER = 21
    CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER = 22
    CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER = 23
    CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER = 24
    CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER = 25
    CUDNN_PARAM_ZDESC = 26
    CUDNN_PARAM_ZDATA_PLACEHOLDER = 27
    CUDNN_PARAM_BN_Z_EQSCALEBIAS_DESC = 28
    CUDNN_PARAM_BN_Z_EQSCALE_PLACEHOLDER = 29
    CUDNN_PARAM_BN_Z_EQBIAS_PLACEHOLDER = 30
    CUDNN_PARAM_ACTIVATION_BITMASK_DESC = 31
    CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER = 32
    CUDNN_PARAM_DXDESC = 33
    CUDNN_PARAM_DXDATA_PLACEHOLDER = 34
    CUDNN_PARAM_DZDESC = 35
    CUDNN_PARAM_DZDATA_PLACEHOLDER = 36
    CUDNN_PARAM_BN_DSCALE_PLACEHOLDER = 37
    CUDNN_PARAM_BN_DBIAS_PLACEHOLDER = 38

    # cudnnFusedOpsPointerPlaceHolder_t
    CUDNN_PTR_NULL = 0
    CUDNN_PTR_ELEM_ALIGNED = 1
    CUDNN_PTR_16B_ALIGNED = 2

    # cudnnFusedOpsVariantParamLabel_t
    CUDNN_PTR_XDATA = 0
    CUDNN_PTR_BN_EQSCALE = 1
    CUDNN_PTR_BN_EQBIAS = 2
    CUDNN_PTR_WDATA = 3
    CUDNN_PTR_DWDATA = 4
    CUDNN_PTR_YDATA = 5
    CUDNN_PTR_DYDATA = 6
    CUDNN_PTR_YSUM = 7
    CUDNN_PTR_YSQSUM = 8
    CUDNN_PTR_WORKSPACE = 9
    CUDNN_PTR_BN_SCALE = 10
    CUDNN_PTR_BN_BIAS = 11
    CUDNN_PTR_BN_SAVED_MEAN = 12
    CUDNN_PTR_BN_SAVED_INVSTD = 13
    CUDNN_PTR_BN_RUNNING_MEAN = 14
    CUDNN_PTR_BN_RUNNING_VAR = 15
    CUDNN_PTR_ZDATA = 16
    CUDNN_PTR_BN_Z_EQSCALE = 17
    CUDNN_PTR_BN_Z_EQBIAS = 18
    CUDNN_PTR_ACTIVATION_BITMASK = 19
    CUDNN_PTR_DXDATA = 20
    CUDNN_PTR_DZDATA = 21
    CUDNN_PTR_BN_DSCALE = 22
    CUDNN_PTR_BN_DBIAS = 23
    CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES = 100
    CUDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT = 101
    CUDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR = 102
    CUDNN_SCALAR_DOUBLE_BN_EPSILON = 103

    # I fished these values out of the cudnn headers
    CUDNN_RNN_NO_BIAS = 0
    CUDNN_RNN_SINGLE_INP_BIAS = 1
    CUDNN_RNN_DOUBLE_BIAS = 2
    CUDNN_RNN_SINGLE_REC_BIAS = 3

    CUDNN_FWD_MODE_INFERENCE = 0
    CUDNN_FWD_MODE_TRAINING = 1

    CUDNN_WGRAD_MODE_ADD = 0
    CUDNN_WGRAD_MODE_SET = 1


cdef extern from './cupy_cudnn.h' nogil:
    # Types
    ctypedef int ActivationMode 'cudnnActivationMode_t'
    ctypedef int AddMode 'cudnnAddMode_t'
    ctypedef int BatchNormMode 'cudnnBatchNormMode_t'
    ctypedef int BatchNormOps 'cudnnBatchNormOps_t'
    ctypedef int ConvolutionBwdDataAlgo 'cudnnConvolutionBwdDataAlgo_t'
    ctypedef int ConvolutionBwdDataPreference \
        'cudnnConvolutionBwdDataPreference_t'
    ctypedef struct ConvolutionBwdDataAlgoPerf \
        'cudnnConvolutionBwdDataAlgoPerf_t':  # NOQA: E125
        int algo
        int status
        float time
        size_t memory
    ctypedef struct ConvolutionBwdDataAlgoPerf_v7 \
        'cudnnConvolutionBwdDataAlgoPerf_v7_t':  # NOQA: E125
        int algo
        int status
        float time
        size_t memory
        int determinism
        int mathType
    ctypedef int ConvolutionBwdFilterAlgo 'cudnnConvolutionBwdFilterAlgo_t'
    ctypedef int ConvolutionBwdFilterPreference \
        'cudnnConvolutionBwdFilterPreference_t'
    ctypedef struct ConvolutionBwdFilterAlgoPerf \
        'cudnnConvolutionBwdFilterAlgoPerf_t':  # NOQA: E125
        int algo
        int status
        float time
        size_t memory
    ctypedef struct ConvolutionBwdFilterAlgoPerf_v7 \
        'cudnnConvolutionBwdFilterAlgoPerf_v7_t':  # NOQA: E125
        int algo
        int status
        float time
        size_t memory
        int determinism
        int mathType
    ctypedef int ConvolutionFwdAlgo 'cudnnConvolutionFwdAlgo_t'
    ctypedef int ConvolutionFwdPreference 'cudnnConvolutionFwdPreference_t'
    ctypedef struct ConvolutionFwdAlgoPerf 'cudnnConvolutionFwdAlgoPerf_t':
        int algo
        int status
        float time
        size_t memory
    ctypedef struct ConvolutionFwdAlgoPerf_v7 \
        'cudnnConvolutionFwdAlgoPerf_v7_t':  # NOQA: E125
        int algo
        int status
        float time
        size_t memory
        int determinism
        int mathType
    ctypedef int cudnnConvolutionMode_t 'cudnnConvolutionMode_t'
    ctypedef int cudnnDataType_t 'cudnnDataType_t'
    ctypedef int cudnnMathType_t 'cudnnMathType_t'
    ctypedef int cudnnDirectionMode_t 'cudnnDirectionMode_t'
    ctypedef int cudnnNanPropagation_t 'cudnnNanPropagation_t'
    ctypedef int cudnnPoolingMode_t 'cudnnPoolingMode_t'
    ctypedef int cudnnRNNInputMode_t 'cudnnRNNInputMode_t'
    ctypedef int cudnnCTCLossAlgo_t 'cudnnCTCLossAlgo_t'
    ctypedef int cudnnRNNMode_t 'cudnnRNNMode_t'
    ctypedef int cudnnRNNAlgo_t 'cudnnRNNAlgo_t'
    ctypedef int cudnnRNNDataLayout_t 'cudnnRNNDataLayout_t'
    ctypedef int cudnnRNNPaddingMode_t 'cudnnRNNPaddingMode_t'
    ctypedef int cudnnSoftmaxAlgorithm_t 'cudnnSoftmaxAlgorithm_t'
    ctypedef int cudnnSoftmaxMode_t 'cudnnSoftmaxMode_t'
    ctypedef int cudnnStatus_t 'cudnnStatus_t'
    ctypedef int cudnnTensorFormat_t 'cudnnTensorFormat_t'
    ctypedef int cudnnOpTensorOp_t 'cudnnOpTensorOp_t'
    ctypedef int cudnnReduceTensorOp_t 'cudnnReduceTensorOp_t'
    ctypedef int cudnnReduceTensorIndices_t 'cudnnReduceTensorIndices_t'
    ctypedef int cudnnIndicesType_t 'cudnnIndicesType_t'
    ctypedef int cudnnErrQueryMode_t 'cudnnErrQueryMode_t'
    ctypedef int cudnnFusedOps_t 'cudnnFusedOps_t'
    ctypedef int cudnnFusedOpsConstParamLabel_t 'cudnnFusedOpsConstParamLabel_t'
    ctypedef int cudnnFusedOpsPointerPlaceHolder_t 'cudnnFusedOpsPointerPlaceHolder_t'
    ctypedef int cudnnFusedOpsVariantParamLabel_t 'cudnnFusedOpsVariantParamLabel_t'
    ctypedef int cudnnWgradMode_t "cudnnWgradMode_t"
    ctypedef struct cudnnRuntimeTag_t 'cudnnRuntimeTag_t'

    ctypedef void* cudnnActivationDescriptor_t 'cudnnActivationDescriptor_t'
    ctypedef void* cudnnConvolutionDescriptor_t 'cudnnConvolutionDescriptor_t'
    ctypedef void* cudnnDropoutDescriptor_t 'cudnnDropoutDescriptor_t'
    ctypedef void* cudnnFilterDescriptor_t 'cudnnFilterDescriptor_t'
    ctypedef void* cudnnHandle_t 'cudnnHandle_t'
    ctypedef void* cudnnPoolingDescriptor_t 'cudnnPoolingDescriptor_t'
    ctypedef void* cudnnCTCLossDescriptor_t 'cudnnCTCLossDescriptor_t'
    ctypedef void* cudnnRNNDescriptor_t 'cudnnRNNDescriptor_t'
    ctypedef void* cudnnRNNDataDescriptor_t 'cudnnRNNDataDescriptor_t'
    ctypedef void* cudnnPersistentRNNPlan_t 'cudnnPersistentRNNPlan_t'
    ctypedef void* cudnnTensorDescriptor_t 'cudnnTensorDescriptor_t'
    ctypedef void* cudnnOpTensorDescriptor_t 'cudnnOpTensorDescriptor_t'
    ctypedef void* cudnnReduceTensorDescriptor_t 'cudnnReduceTensorDescriptor_t'
    ctypedef void* cudnnSpatialTransformerDescriptor_t \
        'cudnnSpatialTransformerDescriptor_t'
    ctypedef void* cudnnSamplerType_t 'cudnnSamplerType_t'
    ctypedef void* cudnnFusedOpsConstParamPack_t 'cudnnFusedOpsConstParamPack_t'
    ctypedef void* cudnnFusedOpsVariantParamPack_t 'cudnnFusedOpsVariantParamPack_t'
    ctypedef void* cudnnFusedOpsPlan_t 'cudnnFusedOpsPlan_t'
    ctypedef void* cudnnRNNBiasMode_t 'cudnnRNNBiasMode_t'
    
    ctypedef void* cudnnSeqDataAxis_t 'cudnnSeqDataAxis_t'
    ctypedef void* cudnnSeqDataDescriptor_t 'cudnnSeqDataDescriptor_t'
    ctypedef void* cudnnForwardMode_t 'cudnnForwardMode_t'

    # Error handling
    const char* cudnnGetErrorString(cudnnStatus_t status)

    # Version
    size_t cudnnGetVersion()

    # Runtime error checking
    int cudnnQueryRuntimeError(cudnnHandle_t handle, cudnnStatus_t *rstatus,
                               cudnnErrQueryMode_t mode, cudnnRuntimeTag_t *tag)

    # Initialization and CUDA cooperation
    int cudnnCreate(cudnnHandle_t* handle)
    int cudnnDestroy(cudnnHandle_t handle)
    int cudnnSetStream(cudnnHandle_t handle, driver.Stream stream)
    int cudnnGetStream(cudnnHandle_t handle, driver.Stream* stream)

    # Tensor manipulation
    int cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* descriptor)
    int cudnnSetTensor4dDescriptor(
        cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format,
        cudnnDataType_t dataType, int n, int c, int h, int w)
    int cudnnSetTensor4dDescriptorEx(
        cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType,
        int n, int c, int h, int w,
        int nStride, int cStride, int hStride, int wStride)
    int cudnnGetTensor4dDescriptor(
        cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t* dataType,
        int* n, int* c, int* h, int* w,
        int* nStride, int* cStride, int* hStride, int* wStride)
    int cudnnSetTensorNdDescriptor(
        cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType, int nbDims,
        int* dimA, int* strideA)
    int cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc)
    int cudnnAddTensor_v3(
        cudnnHandle_t handle, void* alpha, cudnnTensorDescriptor_t bDesc,
        void* b, void* beta, cudnnTensorDescriptor_t y_desc, void* y)

    int cudnnSetSeqDataDescriptor(
	    cudnnSeqDataDescriptor_t seqDataDesc,
        cudnnDataType_t dataType,
	    int nbDims,
	    const int dimA[],
	    const cudnnSeqDataAxis_t axes[],
	    size_t seqLengthArraySize,
	    const int seqLengthArray[],
	    void *paddingFill)

    # Tensor operations
    int cudnnCreateOpTensorDescriptor(cudnnOpTensorDescriptor_t* opTensorDesc)
    int cudnnSetOpTensorDescriptor(
        cudnnOpTensorDescriptor_t opTensorDesc, cudnnOpTensorOp_t opTensorOp,
        cudnnDataType_t opTensorCompType, cudnnNanPropagation_t opTensorNanOpt)
    int cudnnGetOpTensorDescriptor(
        cudnnOpTensorDescriptor_t opTensorDesc, cudnnOpTensorOp_t* opTensorOp,
        cudnnDataType_t* opTensorCompType, cudnnNanPropagation_t* opTensorNanOpt)
    int cudnnDestroyOpTensorDescriptor(cudnnOpTensorDescriptor_t opTensorDesc)
    int cudnnOpTensor(
        cudnnHandle_t handle, cudnnOpTensorDescriptor_t opTensorDesc, void* alpha1,
        cudnnTensorDescriptor_t aDesc, void* A, void* alpha2,
        cudnnTensorDescriptor_t bDesc, void* B, void* beta,
        cudnnTensorDescriptor_t cDesc, void* C)

    # Tensor reductions
    int cudnnCreateReduceTensorDescriptor(
        cudnnReduceTensorDescriptor_t* reduceTensorDesc)
    int cudnnSetReduceTensorDescriptor(
        cudnnReduceTensorDescriptor_t reduceTensorDesc, cudnnReduceTensorOp_t reduceTensorOp,
        cudnnDataType_t reduceTensorCompType, cudnnNanPropagation_t reduceTensorNanOpt,
        cudnnReduceTensorIndices_t reduceTensorIndices,
        cudnnIndicesType_t reduceTensorIndicesType)
    int cudnnGetReduceTensorDescriptor(
        cudnnReduceTensorDescriptor_t reduceTensorDesc,
        cudnnReduceTensorOp_t* reduceTensorOp, cudnnDataType_t* reduceTensorCompType,
        cudnnNanPropagation_t* reduceTensorNanOpt,
        cudnnReduceTensorIndices_t* reduceTensorIndices,
        cudnnIndicesType_t* reduceTensorIndicesType)
    int cudnnDestroyReduceTensorDescriptor(
        cudnnReduceTensorDescriptor_t reduceTensorDesc)
    int cudnnGetReductionIndicesSize(
        cudnnHandle_t handle, cudnnReduceTensorDescriptor_t reduceTensorDesc,
        cudnnTensorDescriptor_t aDesc, cudnnTensorDescriptor_t cDesc, size_t* sizeInBytes)
    int cudnnGetReductionWorkspaceSize(
        cudnnHandle_t handle, cudnnReduceTensorDescriptor_t reduceTensorDesc,
        cudnnTensorDescriptor_t aDesc, cudnnTensorDescriptor_t cDesc, size_t* sizeInBytes)
    int cudnnReduceTensor(
        cudnnHandle_t handle, cudnnReduceTensorDescriptor_t reduceTensorDesc, void* indices,
        size_t indicesSizeInBytes, void* workspace,
        size_t workspaceSizeInBytes, void* alpha, cudnnTensorDescriptor_t aDesc,
        void* A, void* beta, cudnnTensorDescriptor_t cDesc, void* c)
    int cudnnSetTensor(
        cudnnHandle_t handle, cudnnTensorDescriptor_t yDesc, void* y, void* valuePtr)
    int cudnnScaleTensor(
        cudnnHandle_t handle, cudnnTensorDescriptor_t yDesc, void* y, void* alpha)

    # Dropout
    int cudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t* desc)
    int cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc)
    int cudnnDropoutGetStatesSize(cudnnHandle_t handle, size_t* sizeInBytes)
    int cudnnDropoutGetReserveSpaceSize(
        cudnnTensorDescriptor_t xDesc, size_t* sizeInBytes)
    int cudnnSetDropoutDescriptor(
        cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle, float dropout,
        void* states, size_t stateSizeInBytes, unsigned long long seed)
    int cudnnDropoutForward(
        cudnnHandle_t handle, cudnnDropoutDescriptor_t dropoutDesc,
        cudnnTensorDescriptor_t srcDesc, void* srcData,
        cudnnTensorDescriptor_t dstDesc, void* dstData,
        void* reserveSpace, size_t reserveSpaceSizeInBytes)
    int cudnnDropoutBackward(
        cudnnHandle_t handle, cudnnDropoutDescriptor_t dropoutDesc,
        cudnnTensorDescriptor_t dydesc, void* dy, cudnnTensorDescriptor_t dxdesc,
        void* dx, void* reserveSpace, size_t reserveSpaceSizeInBytes)

    # RNN
    int cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t* rnnDesc)

    int cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t rnnDesc)

    int cudnnCreateRNNDataDescriptor(cudnnRNNDataDescriptor_t* RNNDataDesc)

    int cudnnDestroyRNNDataDescriptor(cudnnRNNDataDescriptor_t RNNDataDesc)

    int cudnnSetRNNDescriptor_v8(
	    cudnnRNNDescriptor_t rnnDesc,
	    cudnnRNNAlgo_t algo,
	    cudnnRNNMode_t cellMode,
	    cudnnRNNBiasMode_t biasMode,
	    cudnnDirectionMode_t dirMode,
	    cudnnRNNInputMode_t inputMode,
	    cudnnDataType_t dataType,
	    cudnnDataType_t mathPrec,
	    cudnnMathType_t mathType,
	    int32_t inputSize,
	    int32_t hiddenSize,
	    int32_t projSize,
	    int32_t numLayers,
        cudnnDropoutDescriptor_t dropoutDesc,
        uint32_t auxFlags)

    int cudnnSetRNNDataDescriptor(
        cudnnRNNDataDescriptor_t RNNDataDesc, cudnnDataType_t dataType, cudnnRNNDataLayout_t layout,
        int maxSeqLength, int batchSize, int vectorSize,
        const int seqLengthArray[], void *paddingFill)

    int cudnnGetRNNDataDescriptor(
        cudnnRNNDataDescriptor_t RNNDataDesc, cudnnDataType_t* dataType,
        cudnnRNNDataLayout_t* layout, int* maxSeqLength, int* batchSize,
        int* vectorSize, int arrayLengthRequested, int seqLengthArray[],
        void* paddingFill)

    int cudnnGetRNNWorkspaceSize(
        cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, int seqLength,
        cudnnTensorDescriptor_t* xDesc, size_t* sizeInBytes)

    int cudnnGetRNNTrainingReserveSize(
        cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, int seqLength,
        cudnnTensorDescriptor_t* xDesc, size_t* sizeInBytes)

    int cudnnGetRNNParamsSize(
        cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, cudnnTensorDescriptor_t xDesc,
        size_t* sizeInBytes, cudnnDataType_t dataType)

    int cudnnGetRNNLinLayerMatrixParams(
        cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, int layer,
        cudnnTensorDescriptor_t xDesc, cudnnFilterDescriptor_t wDesc, void* w,
        int linLayerID, cudnnFilterDescriptor_t linLayerMatDesc,
        void** linLayerMat)

    int cudnnGetRNNLinLayerBiasParams(
        cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, int layer,
        cudnnTensorDescriptor_t xDesc, cudnnFilterDescriptor_t wDesc, void* w,
        int linLayerID, cudnnFilterDescriptor_t linLayerBiasDesc,
        void** linLayerBias)

    cudnnStatus_t cudnnRNNForward(
        cudnnHandle_t handle,
        cudnnRNNDescriptor_t rnnDesc,
        cudnnForwardMode_t fwdMode,
        const int32_t devSeqLengths[],
        cudnnRNNDataDescriptor_t xDesc,
        const void *x,
        cudnnRNNDataDescriptor_t yDesc,
        void *y,
        cudnnTensorDescriptor_t hDesc,
        const void *hx,
        void *hy,
        cudnnTensorDescriptor_t cDesc,
        const void *cx,
        void *cy,
        size_t weightSpaceSize,
        const void *weightSpace,
        size_t workSpaceSize,
        void *workSpace,
        size_t reserveSpaceSize,
        void *reserveSpace)
        
    cudnnStatus_t cudnnRNNBackwardData_v8(
        cudnnHandle_t handle,
        cudnnRNNDescriptor_t rnnDesc,
        const int32_t devSeqLengths[],
        cudnnRNNDataDescriptor_t yDesc,
        const void *y,
        const void *dy,
        cudnnRNNDataDescriptor_t xDesc,
        void *dx,
        cudnnTensorDescriptor_t hDesc,
        const void *hx,
        const void *dhy,
        void *dhx,
        cudnnTensorDescriptor_t cDesc,
        const void *cx,
        const void *dcy,
        void *dcx,
        size_t weightSpaceSize,
        const void *weightSpace,
        size_t workSpaceSize,
        void *workSpace,
        size_t reserveSpaceSize,
        void *reserveSpace
    )

    cudnnStatus_t cudnnRNNBackwardWeights_v8(
        cudnnHandle_t handle,
        cudnnRNNDescriptor_t rnnDesc,
        cudnnWgradMode_t addGrad,
        const int32_t devSeqLengths[],
        cudnnRNNDataDescriptor_t xDesc,
        const void *x,
        cudnnTensorDescriptor_t hDesc,
        const void *hx,
        cudnnRNNDataDescriptor_t yDesc,
        const void *y,
        size_t weightSpaceSize,
        void *dweightSpace,
        size_t workSpaceSize,
        void *workSpace,
        size_t reserveSpaceSize,
        void *reserveSpace
    )

    # Build-time version
    int CUDNN_VERSION

    # Constants
    double _CUDNN_BN_MIN_EPSILON 'CUDNN_BN_MIN_EPSILON'


cdef void check_status(cudnnStatus_t status) nogil:
    pass


cdef void* create_rnn_descriptor() nogil:
    cdef size_t address = 0
    check_status(cudnnCreateRNNDescriptor(<cudnnRNNDescriptor_t*>&address))
    return <void*>address


cdef void destroy_rnn_descriptor(void* address) nogil:
    check_status(cudnnDestroyRNNDescriptor(<cudnnRNNDescriptor_t>address))


cdef void* create_rnn_data_descriptor() nogil:
    cdef size_t address = 0
    check_status(cudnnCreateRNNDataDescriptor(<cudnnRNNDataDescriptor_t*>&address))
    return <void*>address


cdef void destroy_rnn_data_descriptor(void* address) nogil:
    check_status(cudnnDestroyRNNDataDescriptor(<cudnnRNNDataDescriptor_t>address))


cdef void set_bilstm_descriptor(
	void* rnn_desc,
	cudnnDataType_t data_type,
	cudnnDataType_t math_prec,
	cudnnMathType_t math_type,
	int32_t input_size,
	int32_t hidden_size,
	int32_t proj_size,
	int32_t num_layers,
    void* dropout_desc,
    uint32_t aux_flags
) nogil:
    check_status(cudnnSetRNNDescriptor_v8(
        <cudnnRNNDescriptor_t>rnn_desc,
        <cudnnRNNAlgo_t>CUDNN_RNN_ALGO_STANDARD,
        <cudnnRNNMode_t>CUDNN_LSTM,
        <cudnnRNNBiasMode_t>CUDNN_RNN_SINGLE_INP_BIAS,
        <cudnnDirectionMode_t>CUDNN_BIDIRECTIONAL,
        <cudnnRNNInputMode_t>CUDNN_LINEAR_INPUT,
        data_type,
        math_prec,
        math_type,
        input_size,
        hidden_size,
        proj_size,
        num_layers,
        <cudnnDropoutDescriptor_t>dropout_desc,
        aux_flags
    ))


cdef void set_bilstm_data_descriptor(
    void* address,
    size_t data_type,
    int max_seq_length,
    int batch_size,
    int vector_size,
    const int* seq_length_array,
    const int* padding_fill
) nogil:
    check_status(cudnnSetRNNDataDescriptor(
        <cudnnRNNDataDescriptor_t>address,
        <cudnnDataType_t>data_type,
        CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
        max_seq_length,
        batch_size,
        vector_size,
        seq_length_array,
        padding_fill
    ))


cdef void set_seq_data_descriptor(
    void* seq_data_desc,
    cudnnDataType_t data_type,
	int nb_dims,
	const int* dim_A,
	const void* axes,
	size_t seq_length_array_stride,
	const int* seq_length_array,
	void* padding_fill
) nogil:
    check_status(cudnnSetSeqDataDescriptor(
        <cudnnSeqDataDescriptor_t>seq_data_desc,
        data_type,
        nb_dims,
        dim_A,
        <const cudnnSeqDataAxis_t*>axes,
        seq_length_array_stride,
        seq_length_array,
        padding_fill
    ))


cdef void bilstm_forward_inference(
    const void* handle,
    const void* rnn_desc,
    const int32_t* dev_seq_lengths,
    const void* x_desc,
    const void* x,
    const void* y_desc,
    void* y,
    const void* h_desc,
    const void *hx,
    void *hy,
    const void* c_desc,
    const void *cx,
    void *cy,
    size_t n_weight_bytes,
    const void* weights,
    size_t n_workspace_bytes,
    void* workspace,
    size_t n_reserve_bytes,
    void* reserve
) nogil:
    check_status(cudnnRNNForward(
        handle,
        (<cudnnRNNDescriptor_t*>rnn_desc)[0],
        <cudnnForwardMode_t>CUDNN_FWD_MODE_INFERENCE,
        dev_seq_lengths,
        (<cudnnRNNDataDescriptor_t*>x_desc)[0], x,
        (<cudnnRNNDataDescriptor_t*>y_desc)[0], y,
        (<cudnnRNNDataDescriptor_t*>h_desc)[0], hx, hy,
        (<cudnnRNNDataDescriptor_t*>c_desc)[0], cx, cy,
        n_weight_bytes, weights,
        n_workspace_bytes, workspace,
        n_reserve_bytes, reserve
    ))


cdef void bilstm_forward_training(
    const void* handle,
    const void* rnn_desc,
    const int32_t* dev_seq_lengths,
    const void* x_desc,
    const void* x,
    const void* y_desc,
    void* y,
    const void* h_desc,
    const void *hx,
    void *hy,
    const void* c_desc,
    const void *cx,
    void *cy,
    size_t n_weight_bytes,
    const void* weights,
    size_t n_workspace_bytes,
    void* workspace,
    size_t n_reserve_bytes,
    void* reserve
) nogil:
    check_status(cudnnRNNForward(
        handle,
        (<cudnnRNNDescriptor_t*>rnn_desc)[0],
        <cudnnForwardMode_t>CUDNN_FWD_MODE_TRAINING,
        dev_seq_lengths,
        (<cudnnRNNDataDescriptor_t*>x_desc)[0], x,
        (<cudnnRNNDataDescriptor_t*>y_desc)[0], y,
        (<cudnnRNNDataDescriptor_t*>h_desc)[0], hx, hy,
        (<cudnnRNNDataDescriptor_t*>c_desc)[0], cx, cy,
        n_weight_bytes, weights,
        n_workspace_bytes, workspace,
        n_reserve_bytes, reserve
    ))


cdef void bilstm_backward(
    const void* handle,
    const void* rnn_desc,
    const int32_t* dev_seq_lengths,
    size_t y_desc,
    const void *y,
    const void *dy,
    size_t x_desc,
    const void *x,
    void *dx,
    const void* h_desc,
    const void *hx,
    const void *dhy,
    void* dhx,
    const void* c_desc,
    const void *cx,
    const void *dcy,
    void *dcx,
    size_t n_weight_bytes,
    const void *weights, void* dweights,
    size_t n_workspace_bytes,
    void* workspace,
    size_t n_reserve_bytes,
    void* reserve
) nogil:
    check_status(cudnnRNNBackwardData_v8(
        (<cudnnHandle_t*>handle)[0],
        (<cudnnRNNDescriptor_t*>rnn_desc)[0],
        dev_seq_lengths,
        (<cudnnRNNDataDescriptor_t*>y_desc)[0], y, dy,
        (<cudnnRNNDataDescriptor_t*>x_desc)[0], dx,
        (<cudnnTensorDescriptor_t*>h_desc)[0], hx, dhy, dhx,
        (<cudnnTensorDescriptor_t*>c_desc)[0], cx, dcy, dcx,
        n_weight_bytes, weights,
        n_workspace_bytes, workspace,
        n_reserve_bytes, reserve
    ));
    check_status(cudnnRNNBackwardWeights_v8(
        (<cudnnHandle_t*>handle)[0],
        (<cudnnRNNDescriptor_t*>rnn_desc)[0],
        <cudnnWgradMode_t>CUDNN_WGRAD_MODE_ADD,
        dev_seq_lengths,
        (<cudnnRNNDataDescriptor_t*>x_desc)[0], x,
        (<cudnnRNNDataDescriptor_t*>h_desc)[0], hx,
        (<cudnnRNNDataDescriptor_t*>y_desc)[0], y,
        n_weight_bytes, dweights,
        n_workspace_bytes, workspace,
        n_reserve_bytes, reserve
    ))


"""
cdef void multi_head_attn_forward_inference(
    size_t handle,
    size_t attn_desc,
    int curr_idx,
    const int* lo_win_idx,
	const int* hi_win_idx,
	const int* dev_seq_lengths_QO,
	const int* dev_seq_lengths_KV,
	size_t q_desc,
	const void* queries,
	const void* residuals,
	size_t k_desc,
	const void* keys,
	size_t v_desc,
	const void* values,
	size_t o_desc,
    void *out,
	size_t n_weight_bytes,
	const void* weights,
	size_t n_workspace_bytes,
	void* workspace
) nogil:
    check_status(cudnnMultiHeadAttnForward(
	    <cudnnHandle_t>handle,
	    <const cudnnAttnDescriptor_t>attn_desc,
	    curr_idx,
        lo_win_idx,
        hi_win_idx,
        dev_seq_lengths_QO,
        dev_Seq_lengths_KV,
	    <const cudnnSeqDataDescriptor_t>q_desc, queries, residuals,
	    <const cudnnSeqDataDescriptor_t>k_desc, keys,
	    <const cudnnSeqDataDescriptor_t>v_desc, values,
	    <const cudnnSeqDataDescriptor_t>o_desc, out,
        n_weight_bytes, weights,
        n_workspace_bytes, workspace,
        0, NULL
    ))

cdef void multi_head_attn_forward_training(
    size_t handle,
    size_t attn_desc,
    int curr_idx,
    const int* lo_win_idx,
	const int* hi_win_idx,
	const int* dev_seq_lengths_QO,
	const int* dev_seq_lengths_KV,
	size_t q_desc,
	const void* queries,
	const void* residuals,
	size_t k_desc,
	const void* keys,
	size_t v_desc,
	const void* values,
	size_t o_desc,
    void *out,
	size_t n_weight_bytes,
	const void* weights,
	size_t n_workspace_bytes,
	void* workspace,
    size_t n_reserve_bytes,
	void* reserve
) nogil:
    check_status(cudnnMultiHeadAttnForward(
	    <cudnnHandle_t>handle,
	    <const cudnnAttnDescriptor_t>attn_desc,
	    curr_idx,
        lo_win_idx,
        hi_win_idx,
        dev_seq_lengths_QO,
        dev_Seq_lengths_KV,
	    <const cudnnSeqDataDescriptor_t>q_desc, queries, residuals,
	    <const cudnnSeqDataDescriptor_t>k_desc, keys,
	    <const cudnnSeqDataDescriptor_t>v_desc, values,
	    <const cudnnSeqDataDescriptor_t>o_desc, out,
        n_weight_bytes, weights,
        n_workspace_bytes, workspace,
        n_reserve_bytes, reserve
    ))


cdef void multi_head_attn_backward(
    size_t handle,
    size_t attn_desc,
    const int* lo_win_idx,
    const int* hi_win_idx,
    const int* dev_seq_lengths_DQDO,
    const int* dev_seq_lengths_DQDV,
    size_t do_desc,
    void* dout,
    size_t dq_desc,
    const void* queries,
    size_t dk_desc,
    void* dkeys,
    const void* keys,
    size_t dv_desc,
    void* dvalues,
    const void* values,
    size_t n_weight_bytes,
    void* dweights,
    const void* weights,
    size_t n_workspace_bytes,
    void* workspace,
    size_t n_reserve_bytes,
    void* reserve
) nogil:
    check_status(cudnnMultiHeadAttnBackwardData(
    	<cudnnHandle_t>handle,
	    <const cudnnAttnDescriptor_t>attn_desc,
        lo_win_idx,
        hi_win_idx,
        dev_seq_lengths_DQDO,
        dev_seq_lengths_DKDV,
        <const cudnnSeqDataDescriptor_t>do_desc, dout,
	    <const cudnnSeqDataDescriptor_t>dq_desc, dqueries, queries,
	    <const cudnnSeqDataDescriptor_t>dk_desc, dkeys, keys,
	    <const cudnnSeqDataDescriptor_t>dv_desc, dvalues, values,
	    n_weight_bytes, weights,
        n_workspace_bytes, workspace,
        n_reserve_bytes, reserve
    ))
    check_status(cudnnMultiHeadAttnBackwardWeights(
    	<cudnnHandle_t>handle,
	    <const cudnnAttnDescriptor_t>attn_desc,
        <cudnnWgradMode_t>addGrad,
	    <const cudnnSeqDataDescriptor_t>q_desc, queries,
	    <const cudnnSeqDataDescriptor_t>k_desc, keys,
	    <const cudnnSeqDataDescriptor_t>v_desc, values,
	    <const cudnnSeqDataDescriptor_t>do_desc, dout,
	    n_weight_bytes, weights,
        n_workspace_bytes, workspace,
        n_reserve_bytes, reserve
    ))
"""
