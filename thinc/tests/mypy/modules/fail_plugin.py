from thinc.api import chain, ReLu, reduce_max, Softmax, add, concatenate

bad_model = chain(ReLu(10), reduce_max(), Softmax())

bad_model2 = add(ReLu(10), reduce_max(), Softmax())

bad_model_only_plugin = chain(
    ReLu(10), ReLu(10), ReLu(10), ReLu(10), reduce_max(), Softmax()
)

bad_model_only_plugin2 = add(
    ReLu(10), ReLu(10), ReLu(10), ReLu(10), reduce_max(), Softmax()
)
reveal_type(bad_model_only_plugin2)

bad_model_only_plugin3 = concatenate(
    ReLu(10), ReLu(10), ReLu(10), ReLu(10), reduce_max(), Softmax()
)

reveal_type(bad_model_only_plugin3)
