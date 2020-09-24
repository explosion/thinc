const DEFAULT_TYPE_URL = '/docs/api-types'

const CUSTOM_TYPES = {
    Model: '/docs/api-model#model',
    Shim: '/docs/api-model#shim',
    Ops: '/docs/api-backends#ops',
    NumpyOps: '/docs/api-backends#ops',
    CupyOps: '/docs/api-backends#ops',
    Config: '/docs/api-config#config',
    Ragged: '/docs/api-types#ragged',
    Padded: '/docs/api-types#padded',
    Array: DEFAULT_TYPE_URL,
    Floats1d: DEFAULT_TYPE_URL,
    Floats2d: DEFAULT_TYPE_URL,
    Floats3d: DEFAULT_TYPE_URL,
    Floats4d: DEFAULT_TYPE_URL,
    FloatsNd: DEFAULT_TYPE_URL,
    Ints1d: DEFAULT_TYPE_URL,
    Ints2d: DEFAULT_TYPE_URL,
    Ints3d: DEFAULT_TYPE_URL,
    Ints4d: DEFAULT_TYPE_URL,
    IntsNd: DEFAULT_TYPE_URL,
    RNNState: DEFAULT_TYPE_URL,
    Generator: DEFAULT_TYPE_URL,
    BaseModel: 'https://pydantic-docs.helpmanual.io/usage/models/',
    Doc: 'https://spacy.io/api/doc',
    Device: 'https://docs-cupy.chainer.org/en/stable/reference/generated/cupy.cuda.Device.html',
    Tensor: 'https://pytorch.org/docs/stable/tensors.html',
}

export default CUSTOM_TYPES
