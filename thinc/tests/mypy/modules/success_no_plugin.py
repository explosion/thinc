from thinc.api import chain, ReLu, reduce_max, Softmax, add

good_model = chain(ReLu(10), ReLu(10), Softmax())
reveal_type(good_model)

good_model2 = add(ReLu(10), ReLu(10), Softmax())
reveal_type(good_model2)

bad_model_undetected = chain(ReLu(10), ReLu(10), reduce_max(), Softmax())
reveal_type(bad_model_undetected)

bad_model_undetected2 = add(ReLu(10), ReLu(10), reduce_max(), Softmax())
reveal_type(bad_model_undetected2)
