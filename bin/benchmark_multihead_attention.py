from thinc.neural._classes.multiheaded_attention import AttentionInputs
from thinc.neural._classes.multiheaded_attention import PaddedAttentionInputs
import numpy
import numpy.random
import plac
from timebudget import timebudget
timebudget.report_atexit()  # Generate report when the program exits


def get_lengths(nr_length, mean=50, scale=10):
    lengths = numpy.random.normal(loc=mean, scale=scale, size=nr_length)
    lengths = lengths.astype("i")
    lengths = numpy.clip(lengths, a_min=1, a_max=None)
    return [length for length in lengths]


def get_random_values(lengths, nH, nD):
    data = numpy.random.uniform(-1, 1, (sum(lengths), 3, nH, nD))
    data = data.astype("f")
    return data


def get_attn_inputs(lengths, values, nH, nD):
    return AttentionInputs(values, lengths)


def get_padded_attn_inputs(lengths, values, nH, nD):
    data = numpy.zeros((len(lengths), max(lengths), 3, nH, nD), dtype="f")
    start = 0
    for i, length in enumerate(lengths):
        data[i, :length] = values[start:start+length]
        start += length
    return PaddedAttentionInputs(data, lengths)


@timebudget
def get_attn_ragged(batch):
    return batch.get_attn()

@timebudget
def get_attn_padded(batch):
    return batch.get_attn()

@timebudget
def apply_attn_ragged(batch, attn):
    return batch.apply_attn(attn)

@timebudget
def apply_attn_padded(batch, attn):
    return batch.apply_attn(attn)

@timebudget
def backprop_apply_ragged(d_output, backprop):
    return backprop(d_output)

@timebudget
def backprop_apply_padded(d_output, backprop):
    return backprop(d_output)

@timebudget
def backprop_attn_ragged(d_attn, backprop):
    return backprop(d_attn)

@timebudget
def backprop_attn_padded(d_attn, backprop):
    return backprop(d_attn)


def main(nr_batch=100, nr_length=30, nH=4, nD=128//4):
    numpy.random.seed(0)
    unpadded = []
    padded = []
    for batch in range(nr_batch):
        lengths = get_lengths(nr_length)
        values = get_random_values(lengths, nH, nD)
        unpadded.append(get_attn_inputs(lengths, values, nH, nD))
        padded.append(get_padded_attn_inputs(lengths, values, nH, nD))
    unpadded_pow = 0.
    for batch in unpadded:
        attn, backprop_attn = get_attn_ragged(batch)
        unpadded_pow += (attn*attn).sum()
        output, backprop_apply = apply_attn_ragged(batch, attn)
        _, d_attn = backprop_apply_ragged(output, backprop_apply)
        backprop_attn_ragged(d_attn, backprop_attn)
    padded_pow = 0.
    for batch in padded:
        attn, backprop_attn = get_attn_padded(batch)
        output, backprop_apply = apply_attn_padded(batch, attn)
        _, d_attn = backprop_apply_padded(output, backprop_apply)
        backprop_attn_padded(d_attn, backprop_attn)
        padded_pow += (attn*attn).sum()
    print(unpadded_pow, padded_pow)
    total_words = sum(batch.nN for batch in unpadded)
    print(total_words)

if __name__ == "__main__":
    plac.call(main)
