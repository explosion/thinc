from thinc.api import chain, ReLu, MaxPool, Softmax, add

bad_model = chain(ReLu(10), MaxPool(), Softmax())

bad_model2 = add(ReLu(10), MaxPool(), Softmax())

bad_model_only_plugin = chain(ReLu(10), ReLu(10), ReLu(10), ReLu(10), MaxPool(), Softmax(), ReLu(10))

bad_model_only_plugin2 = add(ReLu(10), ReLu(10), ReLu(10), ReLu(10), MaxPool(), Softmax(), ReLu(10))
