from thinc.api import chain, Relu, reduce_max, Softmax, add, concatenate

bad_model = chain(Relu(10), reduce_max(), Softmax())

bad_model2 = add(Relu(10), reduce_max(), Softmax())

bad_model_only_plugin = chain(
    Relu(10), Relu(10), Relu(10), Relu(10), reduce_max(), Softmax()
)

bad_model_only_plugin2 = add(
    Relu(10), Relu(10), Relu(10), Relu(10), reduce_max(), Softmax()
)
reveal_type(bad_model_only_plugin2)

bad_model_only_plugin3 = concatenate(
    Relu(10), Relu(10), Relu(10), Relu(10), reduce_max(), Softmax()
)

reveal_type(bad_model_only_plugin3)
