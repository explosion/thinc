from thinc.api import chain, Relu, reduce_max, Softmax, add

bad_model = chain(Relu(10), reduce_max(), Softmax())

bad_model2 = add(Relu(10), reduce_max(), Softmax())
