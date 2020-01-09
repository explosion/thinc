from thinc.api import strings2arrays


def test_strings2arrays():
    strings = ["hello", "world"]
    model = strings2arrays()
    Y, backprop = model.begin_update(strings)
    assert len(Y) == len(strings)
    assert backprop([]) == []
