###############################################################################
# Extern
###############################################################################

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
    ctypedef int ConvolutionMode 'cudnnConvolutionMode_t'
    ctypedef int DataType 'cudnnDataType_t'
    ctypedef int MathType 'cudnnMathType_t'
    ctypedef int DirectionMode 'cudnnDirectionMode_t'
    ctypedef int NanPropagation 'cudnnNanPropagation_t'
    ctypedef int PoolingMode 'cudnnPoolingMode_t'
    ctypedef int RNNInputMode 'cudnnRNNInputMode_t'
    ctypedef int CTCLossAlgo 'cudnnCTCLossAlgo_t'
    ctypedef int RNNMode 'cudnnRNNMode_t'
    ctypedef int RNNAlgo 'cudnnRNNAlgo_t'
    ctypedef int RNNDataLayout 'cudnnRNNDataLayout_t'
    ctypedef int RNNPaddingMode 'cudnnRNNPaddingMode_t'
    ctypedef int SoftmaxAlgorithm 'cudnnSoftmaxAlgorithm_t'
    ctypedef int SoftmaxMode 'cudnnSoftmaxMode_t'
    ctypedef int Status 'cudnnStatus_t'
    ctypedef int TensorFormat 'cudnnTensorFormat_t'
    ctypedef int OpTensorOp 'cudnnOpTensorOp_t'
    ctypedef int ReduceTensorOp 'cudnnReduceTensorOp_t'
    ctypedef int ReduceTensorIndices 'cudnnReduceTensorIndices_t'
    ctypedef int IndicesType 'cudnnIndicesType_t'
    ctypedef int ErrQueryMode 'cudnnErrQueryMode_t'
    ctypedef int FusedOps 'cudnnFusedOps_t'
    ctypedef int FusedOpsConstParamLabel 'cudnnFusedOpsConstParamLabel_t'
    ctypedef int FusedOpsPointerPlaceHolder 'cudnnFusedOpsPointerPlaceHolder_t'
    ctypedef int FusedOpsVariantParamLabel 'cudnnFusedOpsVariantParamLabel_t'
    ctypedef struct RuntimeTag 'cudnnRuntimeTag_t'

    ctypedef void* ActivationDescriptor 'cudnnActivationDescriptor_t'
    ctypedef void* ConvolutionDescriptor 'cudnnConvolutionDescriptor_t'
    ctypedef void* DropoutDescriptor 'cudnnDropoutDescriptor_t'
    ctypedef void* FilterDescriptor 'cudnnFilterDescriptor_t'
    ctypedef void* Handle 'cudnnHandle_t'
    ctypedef void* PoolingDescriptor 'cudnnPoolingDescriptor_t'
    ctypedef void* CTCLossDescriptor 'cudnnCTCLossDescriptor_t'
    ctypedef void* RNNDescriptor 'cudnnRNNDescriptor_t'
    ctypedef void* RNNDataDescriptor 'cudnnRNNDataDescriptor_t'
    ctypedef void* PersistentRNNPlan 'cudnnPersistentRNNPlan_t'
    ctypedef void* TensorDescriptor 'cudnnTensorDescriptor_t'
    ctypedef void* OpTensorDescriptor 'cudnnOpTensorDescriptor_t'
    ctypedef void* ReduceTensorDescriptor 'cudnnReduceTensorDescriptor_t'
    ctypedef void* SpatialTransformerDescriptor \
        'cudnnSpatialTransformerDescriptor_t'
    ctypedef void* SamplerType 'cudnnSamplerType_t'
    ctypedef void* FusedOpsConstParamPack 'cudnnFusedOpsConstParamPack_t'
    ctypedef void* FusedOpsVariantParamPack 'cudnnFusedOpsVariantParamPack_t'
    ctypedef void* FusedOpsPlan 'cudnnFusedOpsPlan_t'

    # Error handling
    const char* cudnnGetErrorString(Status status)

    # Version
    size_t cudnnGetVersion()

    # Runtime error checking
    int cudnnQueryRuntimeError(Handle handle, Status *rstatus,
                               ErrQueryMode mode, RuntimeTag *tag)

    # Initialization and CUDA cooperation
    int cudnnCreate(Handle* handle)
    int cudnnDestroy(Handle handle)
    int cudnnSetStream(Handle handle, driver.Stream stream)
    int cudnnGetStream(Handle handle, driver.Stream* stream)

    # Tensor manipulation
    int cudnnCreateTensorDescriptor(TensorDescriptor* descriptor)
    int cudnnSetTensor4dDescriptor(
        TensorDescriptor tensorDesc, TensorFormat format,
        DataType dataType, int n, int c, int h, int w)
    int cudnnSetTensor4dDescriptorEx(
        TensorDescriptor tensorDesc, DataType dataType,
        int n, int c, int h, int w,
        int nStride, int cStride, int hStride, int wStride)
    int cudnnGetTensor4dDescriptor(
        TensorDescriptor tensorDesc, DataType* dataType,
        int* n, int* c, int* h, int* w,
        int* nStride, int* cStride, int* hStride, int* wStride)
    int cudnnSetTensorNdDescriptor(
        TensorDescriptor tensorDesc, DataType dataType, int nbDims,
        int* dimA, int* strideA)
    int cudnnDestroyTensorDescriptor(TensorDescriptor tensorDesc)
    int cudnnAddTensor_v3(
        Handle handle, void* alpha, TensorDescriptor bDesc,
        void* b, void* beta, TensorDescriptor yDesc, void* y)

    # Tensor operations
    int cudnnCreateOpTensorDescriptor(OpTensorDescriptor* opTensorDesc)
    int cudnnSetOpTensorDescriptor(
        OpTensorDescriptor opTensorDesc, OpTensorOp opTensorOp,
        DataType opTensorCompType, NanPropagation opTensorNanOpt)
    int cudnnGetOpTensorDescriptor(
        OpTensorDescriptor opTensorDesc, OpTensorOp* opTensorOp,
        DataType* opTensorCompType, NanPropagation* opTensorNanOpt)
    int cudnnDestroyOpTensorDescriptor(OpTensorDescriptor opTensorDesc)
    int cudnnOpTensor(
        Handle handle, OpTensorDescriptor opTensorDesc, void* alpha1,
        TensorDescriptor aDesc, void* A, void* alpha2,
        TensorDescriptor bDesc, void* B, void* beta,
        TensorDescriptor cDesc, void* C)

    # Tensor reductions
    int cudnnCreateReduceTensorDescriptor(
        ReduceTensorDescriptor* reduceTensorDesc)
    int cudnnSetReduceTensorDescriptor(
        ReduceTensorDescriptor reduceTensorDesc, ReduceTensorOp reduceTensorOp,
        DataType reduceTensorCompType, NanPropagation reduceTensorNanOpt,
        ReduceTensorIndices reduceTensorIndices,
        IndicesType reduceTensorIndicesType)
    int cudnnGetReduceTensorDescriptor(
        ReduceTensorDescriptor reduceTensorDesc,
        ReduceTensorOp* reduceTensorOp, DataType* reduceTensorCompType,
        NanPropagation* reduceTensorNanOpt,
        ReduceTensorIndices* reduceTensorIndices,
        IndicesType* reduceTensorIndicesType)
    int cudnnDestroyReduceTensorDescriptor(
        ReduceTensorDescriptor reduceTensorDesc)
    int cudnnGetReductionIndicesSize(
        Handle handle, ReduceTensorDescriptor reduceTensorDesc,
        TensorDescriptor aDesc, TensorDescriptor cDesc, size_t* sizeInBytes)
    int cudnnGetReductionWorkspaceSize(
        Handle handle, ReduceTensorDescriptor reduceTensorDesc,
        TensorDescriptor aDesc, TensorDescriptor cDesc, size_t* sizeInBytes)
    int cudnnReduceTensor(
        Handle handle, ReduceTensorDescriptor reduceTensorDesc, void* indices,
        size_t indicesSizeInBytes, void* workspace,
        size_t workspaceSizeInBytes, void* alpha, TensorDescriptor aDesc,
        void* A, void* beta, TensorDescriptor cDesc, void* c)
    int cudnnSetTensor(
        Handle handle, TensorDescriptor yDesc, void* y, void* valuePtr)
    int cudnnScaleTensor(
        Handle handle, TensorDescriptor yDesc, void* y, void* alpha)

    # Filter manipulation
    int cudnnCreateFilterDescriptor(FilterDescriptor* filterDesc)
    int cudnnSetFilter4dDescriptor_v4(
        FilterDescriptor filterDesc, DataType dataType,
        TensorFormat format, int k, int c, int h, int w)
    int cudnnSetFilterNdDescriptor_v4(
        FilterDescriptor filterDesc, DataType dataType,
        TensorFormat format, int nbDims, const int filterDimA[])
    int cudnnGetFilterNdDescriptor_v4(
        FilterDescriptor wDesc, int nbDimsRequested, DataType* dataType,
        TensorFormat* format, int* nbDims, int filterDimA[])
    int cudnnDestroyFilterDescriptor(FilterDescriptor filterDesc)

    # Convolution
    int cudnnCreateConvolutionDescriptor(ConvolutionDescriptor* convDesc)
    int cudnnSetConvolutionMathType(
        ConvolutionDescriptor convDesc, MathType mathType)
    int cudnnGetConvolutionMathType(
        ConvolutionDescriptor convDesc, MathType *mathType)
    int cudnnSetConvolutionGroupCount(
        ConvolutionDescriptor convDesc, int groupCount)
    int cudnnGetConvolutionGroupCount(
        ConvolutionDescriptor convDesc, int *groupCount)
    int cudnnSetConvolution2dDescriptor_v4(
        ConvolutionDescriptor convDesc, int pad_h, int pad_w, int u,
        int v, int dilation_h, int dilation_w, ConvolutionMode mode)
    int cudnnSetConvolution2dDescriptor_v5(
        ConvolutionDescriptor convDesc, int pad_h, int pad_w, int u,
        int v, int dilation_h, int dilation_w, ConvolutionMode mode,
        DataType computeType)
    int cudnnSetConvolutionNdDescriptor_v3(
        ConvolutionDescriptor convDesc, int arrayLength, int* padA,
        int* filterStrideA, int* dilationA, ConvolutionMode mode,
        DataType dataType)
    int cudnnDestroyConvolutionDescriptor(ConvolutionDescriptor conDesc)
    int cudnnFindConvolutionForwardAlgorithm(
        Handle handle, TensorDescriptor xDesc, FilterDescriptor wDesc,
        ConvolutionDescriptor convDesc, TensorDescriptor yDesc,
        int requestedAlgoCount, int* returnedAlgoCount,
        ConvolutionFwdAlgoPerf* perfResults)
    int cudnnFindConvolutionForwardAlgorithmEx(
        Handle handle, TensorDescriptor xDesc, void* x,
        FilterDescriptor wDesc, void* w, ConvolutionDescriptor convDesc,
        TensorDescriptor yDesc, void* y, int requestedAlgoCount,
        int* returnedAlgoCount, ConvolutionFwdAlgoPerf* perfResults,
        void* workSpace, size_t workSpaceSizeInBytes)
    int cudnnFindConvolutionForwardAlgorithmEx_v7(
        Handle handle, TensorDescriptor xDesc, void* x,
        FilterDescriptor wDesc, void* w, ConvolutionDescriptor convDesc,
        TensorDescriptor yDesc, void* y, int requestedAlgoCount,
        int* returnedAlgoCount, ConvolutionFwdAlgoPerf_v7* perfResults,
        void* workSpace, size_t workSpaceSizeInBytes)
    int cudnnGetConvolutionForwardAlgorithm_v6(
        Handle handle, TensorDescriptor srcDesc,
        FilterDescriptor filterDesc, ConvolutionDescriptor convDesc,
        TensorDescriptor destDesc, ConvolutionFwdPreference preference,
        size_t memoryLimitInbytes, ConvolutionFwdAlgo* algo)
    int cudnnGetConvolutionForwardAlgorithm_v7(
        Handle handle, TensorDescriptor srcDesc,
        FilterDescriptor filterDesc, ConvolutionDescriptor convDesc,
        TensorDescriptor destDesc, int requestedAlgoCount,
        int* returnedAlgoCount, ConvolutionFwdAlgoPerf_v7* perfResults)
    int cudnnGetConvolutionForwardWorkspaceSize(
        Handle handle, TensorDescriptor srcDesc,
        FilterDescriptor filterDesc, ConvolutionDescriptor convDesc,
        TensorDescriptor destDesc, ConvolutionFwdAlgo algo,
        size_t* sizeInBytes)
    int cudnnConvolutionForward(
        Handle handle, void* alpha, TensorDescriptor srcDesc,
        void* srcData, FilterDescriptor filterDesc, void* filterData,
        ConvolutionDescriptor convDesc, ConvolutionFwdAlgo algo,
        void* workSpace, size_t workSpaceSizeInBytes, void* beta,
        TensorDescriptor destDesc, void* destData)
    int cudnnConvolutionBackwardBias(
        Handle handle, void* alpha,
        TensorDescriptor srcDesc, void* srcData, void* beta,
        TensorDescriptor destDesc, void* destData)
    int cudnnFindConvolutionBackwardFilterAlgorithm(
        Handle handle, TensorDescriptor xDesc, TensorDescriptor dyDesc,
        ConvolutionDescriptor convDesc, FilterDescriptor dwDesc,
        int requestedAlgoCount, int* returnedAlgoCount,
        ConvolutionBwdFilterAlgoPerf* perfResults)
    int cudnnFindConvolutionBackwardFilterAlgorithmEx(
        Handle handle, TensorDescriptor xDesc, void* x,
        TensorDescriptor dyDesc, void* dy, ConvolutionDescriptor convDesc,
        FilterDescriptor dwDesc, void* dw, int requestedAlgoCount,
        int* returnedAlgoCount, ConvolutionBwdFilterAlgoPerf* perfResults,
        void* workSpace, size_t workSpaceSizeInBytes)
    int cudnnFindConvolutionBackwardFilterAlgorithmEx_v7(
        Handle handle, TensorDescriptor xDesc, void* x,
        TensorDescriptor dyDesc, void* dy, ConvolutionDescriptor convDesc,
        FilterDescriptor dwDesc, void* dw, int requestedAlgoCount,
        int* returnedAlgoCount, ConvolutionBwdFilterAlgoPerf_v7* perfResults,
        void* workSpace, size_t workSpaceSizeInBytes)
    int cudnnGetConvolutionBackwardFilterAlgorithm_v6(
        Handle handle, TensorDescriptor srcDesc, TensorDescriptor diffDesc,
        ConvolutionDescriptor convDesc, FilterDescriptor filterDesc,
        ConvolutionBwdFilterPreference preference,
        size_t memoryLimitInbytes, ConvolutionBwdFilterAlgo* algo)
    int cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        Handle handle, TensorDescriptor srcDesc, TensorDescriptor diffDesc,
        ConvolutionDescriptor convDesc, FilterDescriptor gradDesc,
        int requestedAlgoCount, int* returnedAlgoCount,
        ConvolutionBwdFilterAlgoPerf_v7* perfResults)
    int cudnnGetConvolutionBackwardFilterWorkspaceSize(
        Handle handle, TensorDescriptor srcDesc, TensorDescriptor diffDesc,
        ConvolutionDescriptor convDesc, FilterDescriptor filterDesc,
        ConvolutionBwdFilterAlgo algo, size_t* sizeInBytes)
    int cudnnConvolutionBackwardFilter_v3(
        Handle handle, void* alpha,
        TensorDescriptor srcDesc, void* srcData,
        TensorDescriptor diffDesc, void* diffData,
        ConvolutionDescriptor convDesc, ConvolutionBwdFilterAlgo algo,
        void* workSpace, size_t workSpaceSizeInBytes, void* beta,
        FilterDescriptor gradDesc, void* gradData)
    int cudnnGetConvolutionBackwardDataAlgorithm_v6(
        Handle handle, FilterDescriptor filterDesc,
        TensorDescriptor diffDesc,
        ConvolutionDescriptor convDesc, TensorDescriptor gradDesc,
        ConvolutionBwdDataPreference preference,
        size_t memoryLimitInbytes, ConvolutionBwdDataAlgo* algo)
    int cudnnGetConvolutionBackwardDataAlgorithm_v7(
        Handle handle, TensorDescriptor filterDesc, TensorDescriptor diffDesc,
        ConvolutionDescriptor convDesc, FilterDescriptor gradDesc,
        int requestedAlgoCount, int* returnedAlgoCount,
        ConvolutionBwdDataAlgoPerf_v7* perfResults)
    int cudnnFindConvolutionBackwardDataAlgorithm(
        Handle handle, TensorDescriptor wDesc, TensorDescriptor dyDesc,
        ConvolutionDescriptor convDesc, FilterDescriptor dxDesc,
        int requestedAlgoCount, int* returnedAlgoCount,
        ConvolutionBwdDataAlgoPerf* perfResults)
    int cudnnFindConvolutionBackwardDataAlgorithmEx(
        Handle handle, FilterDescriptor wDesc, void* w,
        TensorDescriptor dyDesc, void* dy, ConvolutionDescriptor convDesc,
        TensorDescriptor dxDesc, void* dx, int requestedAlgoCount,
        int* returnedAlgoCount, ConvolutionBwdDataAlgoPerf* perfResults,
        void* workSpace, size_t workSpaceSizeInBytes)
    int cudnnFindConvolutionBackwardDataAlgorithmEx_v7(
        Handle handle, FilterDescriptor wDesc, void* w,
        TensorDescriptor dyDesc, void* dy, ConvolutionDescriptor convDesc,
        TensorDescriptor dxDesc, void* dx, int requestedAlgoCount,
        int* returnedAlgoCount, ConvolutionBwdDataAlgoPerf_v7* perfResults,
        void* workSpace, size_t workSpaceSizeInBytes)
    int cudnnGetConvolutionBackwardDataWorkspaceSize(
        Handle handle, FilterDescriptor filterDesc,
        TensorDescriptor diffDesc,
        ConvolutionDescriptor convDesc, TensorDescriptor gradDesc,
        ConvolutionBwdDataAlgo algo, size_t* sizeInBytes)
    int cudnnConvolutionBackwardData_v3(
        Handle handle, void* alpha,
        FilterDescriptor filterDesc, void* filterData,
        TensorDescriptor diffDesc, void* diffData,
        ConvolutionDescriptor convDesc, ConvolutionBwdDataAlgo algo,
        void* workSpace, size_t workSpaceSizeInBytes, void* beta,
        TensorDescriptor gradDesc, void* gradData)

    # Pooling
    int cudnnCreatePoolingDescriptor(PoolingDescriptor* desc)
    int cudnnSetPooling2dDescriptor_v4(
        PoolingDescriptor poolingDesc, PoolingMode mode,
        NanPropagation maxpoolingNanOpt, int windowHeight, int windowWidth,
        int verticalPadding, int horizontalPadding, int verticalStride,
        int horizontalStride)
    int cudnnSetPoolingNdDescriptor_v4(
        PoolingDescriptor poolingDesc, PoolingMode mode,
        NanPropagation maxpoolingNanOpt, int nbDims,
        int* windowDimA, int* paddingA, int* strideA)
    int cudnnDestroyPoolingDescriptor(PoolingDescriptor poolingDesc)
    int cudnnPoolingForward(
        Handle handle, PoolingDescriptor poolingDesc, void* alpha,
        TensorDescriptor srcDesc, void* srcData, void* beta,
        TensorDescriptor dstDesc, void* dstData)
    int cudnnPoolingBackward(
        Handle handle, PoolingDescriptor poolingDesc, void* alpha,
        TensorDescriptor srcDesc, void* srcData,
        TensorDescriptor srcDiffDesc, void* srcDiffData,
        TensorDescriptor destDesc, void* destData, void* beta,
        TensorDescriptor destDiffDesc, void* destDiffData)

    # Batch Normalization
    int cudnnDeriveBNTensorDescriptor(
        TensorDescriptor derivedBnDesc, TensorDescriptor xDesc,
        BatchNormMode mode)
    int cudnnBatchNormalizationForwardTraining(
        Handle handle, BatchNormMode mode,
        void* alpha, void* beta, TensorDescriptor xDesc,
        void* x, TensorDescriptor yDesc, void* y,
        TensorDescriptor bnScaleBiasMeanVarDesc, void* bnScale,
        void* bnBias, double exponentialAverageFactor,
        void* resultRunningMean, void* resultRunningVariance,
        double epsilon, void* resultSaveMean,
        void* resultSaveInvVariance)
    int cudnnBatchNormalizationForwardInference(
        Handle handle, BatchNormMode mode,
        void* alpha, void* beta, TensorDescriptor xDesc,
        void* x, TensorDescriptor yDesc, void* y,
        TensorDescriptor bnScaleBiasMeanVarDesc, void* bnScale,
        void* bnBias, void* estimatedMean, void* estimatedVariance,
        double epsilon)
    int cudnnBatchNormalizationBackward(
        Handle handle, BatchNormMode mode,
        void* alphaDataDiff, void* betaDataDiff,
        void* alphaParamDiff, void* betaParamDiff,
        TensorDescriptor xDesc, void* x,
        TensorDescriptor dyDesc, void* dy,
        TensorDescriptor dxDesc, void* dx,
        TensorDescriptor dBnScaleBiasDesc, void* bnScale,
        void* dBnScaleResult, void* dBnBiasResult,
        double epsilon, void* savedMean, void* savedInvVariance)

    int cudnnBatchNormalizationForwardTrainingEx(
        Handle handle,
        BatchNormMode mode, BatchNormOps bnOps,
        void* alpha, void* beta,
        TensorDescriptor xDesc, void* x,
        TensorDescriptor zDesc, void* z,
        TensorDescriptor yDesc, void* y,
        TensorDescriptor bnScaleBiasMeanVarDesc,
        void* bnScale, void* bnBias,
        double exponentialAverageFactor,
        void* resultRunningMean, void* resultRunningVariance,
        double epsilon,
        void* resultSaveMean, void* resultSaveInvVariance,
        ActivationDescriptor activationDesc,
        void* workspace, size_t workSpaceSizeInBytes,
        void* reserveSpace, size_t reserveSpaceSizeInBytes)
    int cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
        Handle handle,
        BatchNormMode mode, BatchNormOps bnOps,
        TensorDescriptor xDesc,
        TensorDescriptor zDesc,
        TensorDescriptor yDesc,
        TensorDescriptor bnScaleBiasMeanVarDesc,
        ActivationDescriptor activationDesc,
        size_t* sizeInBytes)
    int cudnnBatchNormalizationBackwardEx(
        Handle handle,
        BatchNormMode mode, BatchNormOps bnops,
        void* alphaDataDiff, void* betaDataDiff,
        void* alphaParamDiff, void* betaParamDiff,
        TensorDescriptor xDesc, void* x,
        TensorDescriptor yDesc, void* y,
        TensorDescriptor dyDesc, void* dy,
        TensorDescriptor dzDesc, void* dz,
        TensorDescriptor dxDesc, void* dx,
        TensorDescriptor dBnScaleBiasDesc,
        void* bnScaleData, void* bnBiasData,
        void* dBnScaleData, void* dBnBiasData,
        double epsilon,
        void* savedMean, void* savedInvVariance,
        ActivationDescriptor activationDesc,
        void* workspace, size_t workSpaceSizeInBytes,
        void* reserveSpace, size_t reserveSpaceSizeInBytes)
    int cudnnGetBatchNormalizationBackwardExWorkspaceSize(
        Handle handle,
        BatchNormMode mode,
        BatchNormOps bnOps,
        TensorDescriptor xDesc,
        TensorDescriptor yDesc,
        TensorDescriptor dyDesc,
        TensorDescriptor dzDesc,
        TensorDescriptor dxDesc,
        TensorDescriptor dBnScaleBiasDesc,
        ActivationDescriptor activationDesc,
        size_t* sizeInBytes)
    int cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
        Handle handle,
        BatchNormMode mode,
        BatchNormOps bnOps,
        ActivationDescriptor activationDesc,
        TensorDescriptor xDesc,
        size_t* sizeInBytes)

    # Activation
    int cudnnCreateActivationDescriptor(
        ActivationDescriptor* activationDesc)
    int cudnnSetActivationDescriptor(
        ActivationDescriptor activationDesc, ActivationMode mode,
        NanPropagation reluNanOpt, double reluCeiling)
    int cudnnDestroyActivationDescriptor(
        ActivationDescriptor activationDesc)
    int cudnnSoftmaxForward(
        Handle handle, SoftmaxAlgorithm algorithm, SoftmaxMode mode,
        void* alpha, TensorDescriptor srcDesc, void* srcData,
        void* beta, TensorDescriptor dstDesc, void* dstData)
    int cudnnSoftmaxBackward(
        Handle handle, SoftmaxAlgorithm algorithm, SoftmaxMode mode,
        void* alpha, TensorDescriptor srcDesc, void* srcData,
        TensorDescriptor srcDiffDesc, void* srcDiffData, void* beta,
        TensorDescriptor destDiffDesc, void* destDiffData)
    int cudnnActivationForward_v4(
        Handle handle, ActivationDescriptor activationDesc, void* alpha,
        TensorDescriptor srcDesc, void* srcData, void* beta,
        TensorDescriptor dstDesc, void* dstData)
    int cudnnActivationBackward_v4(
        Handle handle, ActivationDescriptor activationDesc, void* alpha,
        TensorDescriptor srcDesc, void* srcData,
        TensorDescriptor srcDiffDesc, void* srcDiffData,
        TensorDescriptor destDesc, void* destData, void* beta,
        TensorDescriptor destDiffDesc, void* destDiffData)

    # Dropout
    int cudnnCreateDropoutDescriptor(DropoutDescriptor* desc)
    int cudnnDestroyDropoutDescriptor(DropoutDescriptor dropoutDesc)
    int cudnnDropoutGetStatesSize(Handle handle, size_t* sizeInBytes)
    int cudnnDropoutGetReserveSpaceSize(
        TensorDescriptor xDesc, size_t* sizeInBytes)
    int cudnnSetDropoutDescriptor(
        DropoutDescriptor dropoutDesc, Handle handle, float dropout,
        void* states, size_t stateSizeInBytes, unsigned long long seed)
    int cudnnDropoutForward(
        Handle handle, DropoutDescriptor dropoutDesc,
        TensorDescriptor srcDesc, void* srcData,
        TensorDescriptor dstDesc, void* dstData,
        void* reserveSpace, size_t reserveSpaceSizeInBytes)
    int cudnnDropoutBackward(
        Handle handle, DropoutDescriptor dropoutDesc,
        TensorDescriptor dydesc, void* dy, TensorDescriptor dxdesc,
        void* dx, void* reserveSpace, size_t reserveSpaceSizeInBytes)

    # CTC
    int cudnnCreateCTCLossDescriptor(CTCLossDescriptor* ctcLossDesc)
    int cudnnDestroyCTCLossDescriptor(CTCLossDescriptor ctcLossDesc)
    int cudnnSetCTCLossDescriptor(
        CTCLossDescriptor ctcLossDesc, DataType dataType)
    int cudnnGetCTCLossDescriptor(
        CTCLossDescriptor ctcLossDesc, DataType* dataType)
    int cudnnGetCTCLossWorkspaceSize(
        Handle handle, TensorDescriptor probsDesc,
        TensorDescriptor gradientsDesc, int* labels,
        int* labelLengths, int* inputLengths, CTCLossAlgo algo,
        CTCLossDescriptor ctcLossDesc, size_t* sizeInBytes)
    int cudnnCTCLoss(
        Handle handle, TensorDescriptor probsDesc,
        void* probs, int* labels, int* labelLengths, int* inputLengths,
        void* costs, TensorDescriptor gradientsDesc, void* gradients,
        CTCLossAlgo algo, CTCLossDescriptor ctcLossDesc,
        void* workspace, size_t workSpaceSizeInBytes)
    # RNN
    int cudnnCreateRNNDescriptor(RNNDescriptor* rnnDesc)
    int cudnnDestroyRNNDescriptor(RNNDescriptor rnnDesc)
    int cudnnCreatePersistentRNNPlan(
        RNNDescriptor rnnDesc,
        const int minibatch, DataType dataType,
        PersistentRNNPlan* plan)
    int cudnnSetPersistentRNNPlan(
        RNNDescriptor rnnDesc, PersistentRNNPlan plan)
    int cudnnDestroyPersistentRNNPlan(PersistentRNNPlan plan)
    int cudnnSetRNNDescriptor_v5(
        RNNDescriptor rnnDesc, int hiddenSize,
        int numLayers, DropoutDescriptor dropoutDesc, RNNInputMode inputMode,
        DirectionMode direction, RNNMode mode, DataType dataType)
    int cudnnSetRNNDescriptor_v6(
        Handle handle, RNNDescriptor rnnDesc, int hiddenSize,
        int numLayers, DropoutDescriptor dropoutDesc, RNNInputMode inputMode,
        DirectionMode direction, RNNMode mode, RNNAlgo algo, DataType dataType)
    int cudnnSetRNNPaddingMode(
        RNNDescriptor rnnDesc, RNNPaddingMode paddingMode)
    int cudnnGetRNNPaddingMode(
        RNNDescriptor rnnDesc, RNNPaddingMode* paddingMode)
    int cudnnCreateRNNDataDescriptor(RNNDataDescriptor* RNNDataDesc)
    int cudnnDestroyRNNDataDescriptor(RNNDataDescriptor RNNDataDesc)
    int cudnnSetRNNDataDescriptor(
        RNNDataDescriptor RNNDataDesc, DataType dataType, RNNDataLayout layout,
        int maxSeqLength, int batchSize, int vectorSize,
        const int seqLengthArray[], void *paddingFill)
    int cudnnGetRNNDataDescriptor(
        RNNDataDescriptor RNNDataDesc, DataType* dataType,
        RNNDataLayout* layout, int* maxSeqLength, int* batchSize,
        int* vectorSize, int arrayLengthRequested, int seqLengthArray[],
        void* paddingFill)
    int cudnnGetRNNWorkspaceSize(
        Handle handle, RNNDescriptor rnnDesc, int seqLength,
        TensorDescriptor* xDesc, size_t* sizeInBytes)
    int cudnnGetRNNTrainingReserveSize(
        Handle handle, RNNDescriptor rnnDesc, int seqLength,
        TensorDescriptor* xDesc, size_t* sizeInBytes)
    int cudnnGetRNNParamsSize(
        Handle handle, RNNDescriptor rnnDesc, TensorDescriptor xDesc,
        size_t* sizeInBytes, DataType dataType)
    int cudnnGetRNNLinLayerMatrixParams(
        Handle handle, RNNDescriptor rnnDesc, int layer,
        TensorDescriptor xDesc, FilterDescriptor wDesc, void* w,
        int linLayerID, FilterDescriptor linLayerMatDesc,
        void** linLayerMat)
    int cudnnGetRNNLinLayerBiasParams(
        Handle handle, RNNDescriptor rnnDesc, int layer,
        TensorDescriptor xDesc, FilterDescriptor wDesc, void* w,
        int linLayerID, FilterDescriptor linLayerBiasDesc,
        void** linLayerBias)
    int cudnnRNNForwardInference(
        Handle handle, RNNDescriptor rnnDesc, int seqLength,
        TensorDescriptor* xDesc,
        void* x, TensorDescriptor hxDesc, void* hx, TensorDescriptor cxDesc,
        void* cx, FilterDescriptor wDesc, void* w, TensorDescriptor* yDesc,
        void* y, TensorDescriptor hyDesc, void* hy, TensorDescriptor cyDesc,
        void* cy, void* workspace, size_t workSpaceSizeInBytes)
    int cudnnRNNForwardTraining(
        Handle handle, RNNDescriptor rnnDesc, int seqLength,
        TensorDescriptor* xDesc, void* x,
        TensorDescriptor hxDesc, void* hx, TensorDescriptor cxDesc, void* cx,
        FilterDescriptor wDesc, void* w, TensorDescriptor* yDesc, void* y,
        TensorDescriptor hyDesc, void* hy, TensorDescriptor cyDesc, void* cy,
        void* workspace, size_t workSpaceSizeInBytes, void* reserveSpace,
        size_t reserveSpaceSizeInBytes)
    int cudnnRNNBackwardData(
        Handle handle, RNNDescriptor rnnDesc, int seqLength,
        TensorDescriptor* yDesc, void* y,
        TensorDescriptor* dyDesc, void* dy,
        TensorDescriptor dhyDesc, void* dhy,
        TensorDescriptor dcyDesc, void* dcy,
        FilterDescriptor wDesc, void* w,
        TensorDescriptor hxDesc, void* hx,
        TensorDescriptor cxDesc, void* cx,
        TensorDescriptor* dxDesc, void* dx,
        TensorDescriptor dhxDesc, void* dhx,
        TensorDescriptor dcxDesc, void* dcx, void* workspace,
        size_t workSpaceSizeInBytes, void* reserveSpace,
        size_t reserveSpaceSizeInBytes)
    int cudnnRNNBackwardWeights(
        Handle handle, RNNDescriptor rnnDesc, int seqLength,
        TensorDescriptor* xDesc, void* x, TensorDescriptor hxDesc, void* hx,
        TensorDescriptor* yDesc, void* y,
        void* workspace, size_t workSpaceSizeInBytes, FilterDescriptor dwDesc,
        void* dw, void* reserveSpace, size_t reserveSpaceSizeInBytes)

    int cudnnRNNForwardInferenceEx(
        Handle handle, RNNDescriptor rnnDesc,
        RNNDataDescriptor xDesc, const void* x,
        TensorDescriptor hxDesc, const void* hx,
        TensorDescriptor cxDesc, const void* cx,
        FilterDescriptor wDesc, const void* w,
        RNNDataDescriptor yDesc, void* y,
        TensorDescriptor hyDesc, void* hy,
        TensorDescriptor cyDesc, void* cy,
        RNNDataDescriptor kDesc, const void* keys,
        RNNDataDescriptor cDesc, void* cAttn,
        RNNDataDescriptor iDesc, void* iAttn,
        RNNDataDescriptor qDesc, void* queries,
        void* workSpace, size_t workSpaceSizeInBytes)
    int cudnnRNNForwardTrainingEx(
        Handle handle, RNNDescriptor rnnDesc,
        RNNDataDescriptor xDesc, const void* x,
        TensorDescriptor hxDesc, const void* hx,
        TensorDescriptor cxDesc, const void* cx,
        FilterDescriptor wDesc, const void* w,
        RNNDataDescriptor yDesc, void* y,
        TensorDescriptor hyDesc, void* hy,
        TensorDescriptor cyDesc, void* cy,
        RNNDataDescriptor kDesc, const void* keys,
        RNNDataDescriptor cDesc, void* cAttn,
        RNNDataDescriptor iDesc, void* iAttn,
        RNNDataDescriptor qDesc, void* queries,
        void* workSpace, size_t workSpaceSizeInBytes,
        void* reserveSpace, size_t reserveSpaceSizeInBytes)
    int cudnnRNNBackwardDataEx(
        Handle handle, RNNDescriptor rnnDesc,
        RNNDataDescriptor yDesc, const void* y,
        RNNDataDescriptor dyDesc, const void* dy,
        RNNDataDescriptor dcDesc, const void* dcAttn,
        TensorDescriptor dhyDesc, const void* dhy,
        TensorDescriptor dcyDesc, const void* dcy,
        FilterDescriptor wDesc, const void* w,
        TensorDescriptor hxDesc, const void* hx,
        TensorDescriptor cxDesc, const void* cx,
        RNNDataDescriptor dxDesc, void* dx,
        TensorDescriptor dhxDesc, void* dhx,
        TensorDescriptor dcxDesc, void* dcx,
        RNNDataDescriptor dkDesc, void* dkeys,
        void* workSpace, size_t workSpaceSizeInBytes,
        void* reserveSpace, size_t reserveSpaceSizeInBytes)
    int cudnnRNNBackwardWeightsEx(
        Handle handle, RNNDescriptor rnnDesc,
        RNNDataDescriptor xDesc, const void* x,
        TensorDescriptor hxDesc, const void* hx,
        RNNDataDescriptor yDesc, const void* y,
        void* workSpace, size_t workSpaceSizeInBytes,
        FilterDescriptor dwDesc, void* dw,
        void* reserveSpace, size_t reserveSpaceSizeInBytes)

    # Spatial Transformer
    int cudnnCreateSpatialTransformerDescriptor(
        SpatialTransformerDescriptor* stDesc)
    int cudnnDestroySpatialTransformerDescriptor(
        SpatialTransformerDescriptor stDesc)
    int cudnnSetSpatialTransformerNdDescriptor(
        SpatialTransformerDescriptor stDesc, SamplerType samplerType,
        DataType dataType, int nbDims, int dimA[])
    int cudnnSpatialTfGridGeneratorForward(
        Handle handle, SpatialTransformerDescriptor stDesc,
        void* theta, void* grid)
    int cudnnSpatialTfGridGeneratorBackward(
        Handle handle, SpatialTransformerDescriptor stDesc,
        void* dgrid, void* dtheta)
    int cudnnSpatialTfSamplerForward(
        Handle handle, SpatialTransformerDescriptor stDesc,
        void* alpha, TensorDescriptor xDesc, void* x,
        void* grid, void* beta, TensorDescriptor yDesc, void* y)
    int cudnnSpatialTfSamplerBackward(
        Handle handle, SpatialTransformerDescriptor stDesc,
        void* alpha, TensorDescriptor xDesc, void* x, void* beta,
        TensorDescriptor dxDesc, void* dx, void* alphaDgrid,
        TensorDescriptor dyDesc, void* dy, void* grid,
        void* betaDgrid, void* dgrid)

    # Fused Ops
    int cudnnCreateFusedOpsConstParamPack(
        FusedOpsConstParamPack* constPack, int ops)
    int cudnnDestroyFusedOpsConstParamPack(FusedOpsConstParamPack constPack)
    int cudnnSetFusedOpsConstParamPackAttribute(
        FusedOpsConstParamPack constPack, FusedOpsConstParamLabel paramLabel,
        const void *param)
    int cudnnGetFusedOpsConstParamPackAttribute(
        const FusedOpsConstParamPack constPack,
        FusedOpsConstParamLabel paramLabel, void *param, int *isNULL)
    int cudnnCreateFusedOpsVariantParamPack(
        FusedOpsVariantParamPack *varPack, FusedOps ops)
    int cudnnDestroyFusedOpsVariantParamPack(FusedOpsVariantParamPack varPack)
    int cudnnSetFusedOpsVariantParamPackAttribute(
        FusedOpsVariantParamPack varPack, FusedOpsVariantParamLabel paramLabel,
        void *ptr)
    int cudnnGetFusedOpsVariantParamPackAttribute(
        const FusedOpsVariantParamPack varPack,
        FusedOpsVariantParamLabel paramLabel, void *ptr)
    int cudnnCreateFusedOpsPlan(FusedOpsPlan *plan, FusedOps ops)
    int cudnnDestroyFusedOpsPlan(FusedOpsPlan plan)
    int cudnnMakeFusedOpsPlan(
        Handle handle, FusedOpsPlan plan,
        const FusedOpsConstParamPack constPack, size_t *workspaceSizeInBytes)
    int cudnnFusedOpsExecute(
        Handle handle, const FusedOpsPlan plan,
        FusedOpsVariantParamPack varPack)

    # Build-time version
    int CUDNN_VERSION

    # Constants
    double _CUDNN_BN_MIN_EPSILON 'CUDNN_BN_MIN_EPSILON'


cdef void check_status(status_t status) nogil:
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
        CUDNN_RNN_ALGO_STANDARD,
        CUDNN_RNN_CELL_MODE_LSTM,
        CUDNN_RNN_SINGLE_INP_BIAS,
        CUDNN_BIDIRECTIONAL,
        CUDNN_LINEAR_INPUT,
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
        <cudnnSeqDataDescriptor_t>seqDataDesc,
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
    const void weights,
    size_t n_workspace_bytes,
    void* workspace,
    size_t n_reserve_bytes,
    void* reserve
) nogil:
    check_status(cudnnRNNForward(
        handle,
        (<cudnnRNNDescriptor_t*>rnn_desc)[0],
        CUDNN_FWD_MODE_INFERENCE,
        dev_seq_lengths,
        (<cudnnRNNDataDescriptor_t*>x_desc)[0], x,
        (<cudnnRNNDataDescriptor_t*>yDesc)[0], y,
        (<cudnnRNNDataDescriptor_t*>h_desc)[0], hx, hy,
        (<cudnnRNNDataDescriptor_t*>c_desc)[0], cx, cy,
        n_weight_bytes, weights,
        n_workspace_bytes, workspace,
        n_reserve_bytes, reserve_space
    ))


cdef void bilstm_forward_training(
    const void* handle,
    const void* rnn_desc,
    const int32_t* dev_seq_lengths,
 
) nogil:
    check_status(cudnnRNNForward(
        handle,
        (<cudnnRNNDescriptor_t*>rnn_desc)[0],
        CUDNN_FWD_MODE_TRAINING,
        dev_seq_lengths,
        (<cudnnRNNDataDescriptor_t*>x_desc)[0], x,
        (<cudnnRNNDataDescriptor_t*>yDesc)[0], y,
        (<cudnnRNNDataDescriptor_t*>h_desc)[0], hx, hy,
        (<cudnnRNNDataDescriptor_t*>c_desc)[0], cx, cy,
        n_weight_bytes, weights,
        n_workspace_bytes, workspace,
        n_reserve_bytes, reserve_space
    ))


cdef void bilstm_backward(
    const void* handle,
    const void* rnn_desc,
    const int32_t* dev_seq_lengths,
    size_t y_desc,
    const void *y,
    const void *dy,
    size_t x_desc,
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
    const void *weights,
    size_t n_workspace_bytes,
    void* workspace,
    size_t n_reserve_bytes,
    void* reserve
) nogil:
    check_status(cudnnRNNBackwardData_v8(
        (<cudnnHandle_t*>handle)[0],
        (<cudnnRNNDescriptor_t*>rnnDesc)[0],
        dev_seq_lengths,
        (<cudnnRNNDataDescriptor_t*>y_desc)[0], y, dy,
        (<cudnnRNNDataDescriptor_t*>x_desc)[0], dx,
        (<cudnnTensorDescriptor_t*>h_desc)[0], hx, dhy, dhx,
        (<cudnnTensorDescriptor_t*>c_desc)[0], cx, dcy, dcx,
        n_weight_bytes, weights,
        n_workspace_bytes, workspace,
        n_reserve_bytes, reserve
    ));
    check_status(cudnnRNNBackwardData_v8(
        (<cudnnHandle_t*>handle)[0],
        (<cudnnRNNDescriptor_t*>rnnDesc)[0],
        dev_seq_lengths,
        (<cudnnRNNDataDescriptor_t*>y_desc)[0], y, dy,
        (<cudnnRNNDataDescriptor_t*>x_desc)[0], x, dx,
        (<cudnnTensorDescriptor_t*>h_desc)[0], dx, dhy, dhx,
        (<cudnnTensorDescriptor_t*>cDesc)[0], cx, dcy, dcx,
        weightSpaceSize, weightSpace,
        workSpaceSize, workSpace,
        reserveSpaceSize, reserveSpace
    ))


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
