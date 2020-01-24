from thinc.api import chain, ReLu, reduce_max, Softmax, add

bad_model = chain(ReLu(10), reduce_max(), Softmax())

bad_model2 = add(ReLu(10), reduce_max(), Softmax())
