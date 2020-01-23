import random
from typing import Dict, List, Optional, Set, Tuple
from wasabi import msg
import numpy
from thinc.api import TensorFlowWrapper, Adam, Model, Ops, get_current_ops
from thinc.types import Array2d, Array
from thinc.util import to_categorical

from mathy import (
    BaseRule,
    ExpressionParser,
    MathExpression,
    TermEx,
    get_term_ex,
    get_terms,
)
from mathy.problems import gen_simplify_multiple_terms, mathy_term_string

from typing import List
import thinc
from thinc.api import (
    Model,
    chain,
    list2array,
    list2padded,
    list2ragged,
    padded2list,
    with_array,
    with_ragged,
    ragged2list,
    reduce_sum,
    LayerNorm,
    with_list,
    Maxout,
    reduce_mean,
    Dropout,
    ParametricAttention,
    Softmax,
    Embed,
    ReLu,
)
from thinc.types import Array2d, Array1d, Padded


parser = ExpressionParser()
Example = Array2d
Label = int
vocab = " .+-/^*()[]-01234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def to_example(input_problem: str) -> Tuple[Example, Label]:
    """Convert an input polynomial expression into a an Example/Label.
    The output is a one-hot encoded input string and a scalar label
    indicating the total number of like terms in the generated expression."""

    expression: MathExpression = parser.parse(input_problem)
    term_nodes: List[MathExpression] = get_terms(expression)
    BaseRule().find_nodes(expression)  # sets node.r_index
    node_groups: Dict[str, List[int]] = {}
    max_index = 0
    for term_node in term_nodes:
        max_index = max(max_index, term_node.r_index)
        ex: Optional[TermEx] = get_term_ex(term_node)
        assert ex is not None, f"invalid expression {ex}"
        key = mathy_term_string(variable=ex.variable, exponent=ex.exponent)
        if key == "":
            key = "const"
        if key not in node_groups:
            node_groups[key] = [term_node]
        else:
            node_groups[key].append(term_node)

    like_terms = 0
    for k, v in node_groups.items():
        if len(v) <= 1:
            continue
        like_terms += len(v)

    eg = to_categorical(
        numpy.array([vocab.index(s) for s in input_problem]), n_classes=len(vocab)
    )

    return eg, like_terms


def generate_dataset(
    size: int, exclude: Optional[Set[str]] = None
) -> Tuple[List[str], List[Example], List[Label]]:
    texts: List[str] = []
    examples: List[Example] = []
    labels: List[Label] = []
    if exclude is None:
        exclude = set()
    skips = 0
    skip_max = size // 2  # if half of the requested number are duplicates, give up
    while len(texts) < size:
        text, complexity = gen_simplify_multiple_terms(
            random.randint(2, 8), noise_probability=1.0, noise_terms=10
        )
        if text in exclude:
            if skips > skip_max:
                raise ValueError(
                    "Encountered more duplicates than allowed. This can be caused "
                    "by a problem generation function that does not do a good job "
                    "of choosing random numbers, operators and variables."
                )
            skips += 1
            continue
        exclude.add(text)
        x, y = to_example(text)
        texts.append(text)
        examples.append(x)
        labels.append(y)

    return texts, examples, labels


def CountLikeTerms(
    n_hidden: int, dropout: float = 0.5
) -> Model[List[Array2d], Array1d]:
    import tensorflow as tf

    # TODO: adding this model causes gradient errors :sweat:
    # tf_module: Model[Array2d, Padded] = TensorFlowWrapper(
    #     tf.keras.Sequential(
    #         [
    #             tf.keras.layers.Dense(n_hidden),
    #             tf.keras.layers.LayerNormalization(),
    #             tf.keras.layers.Dense(1),
    #         ]
    #     )
    # )

    with Model.define_operators({">>": chain}):
        model = (
            list2ragged()
            # TODO: Lots of errors trying to convert things back and forth. Should all layers
            #       be able to deal with conversions between the ragged/list/padded/array types
            #       and the ones they expect? Maybe I'm just doing it wrong?
            # >> with_ragged(
            #     ragged2list()
            #     >> list2array()
            #     >> Maxout(n_hidden)
            #     >> LayerNorm()
            #     >> Dropout(0.2)
            # )
            >> reduce_sum()
            >> ReLu(n_hidden)
            >> Dropout(dropout)
            >> ReLu(n_hidden)
            >> ReLu(n_hidden)
            >> ReLu(n_hidden)
            >> ReLu(1)
        )
    return model


def evaluate_model(model, *, print_problems: bool = False, texts, X, Y):
    Yeval = model.predict(X)
    correct_count = 0
    for text, y_answer, y_guess in zip(texts, Y, Yeval):
        y_guess = round(float(y_guess))
        correct = y_guess == int(y_answer)
        print_fn = msg.fail
        if correct:
            correct_count += 1
            print_fn = msg.good
        if print_problems:
            print_fn(f"Text[{text}] Answer[{y_answer}] Guess[{y_guess}]")
    return correct_count / len(X)


if __name__ == "__main__":
    batch_size = 12
    num_iters = 50
    train_size = 5000
    test_size = 128
    ops: Ops = get_current_ops()
    seen_texts: Set[str] = set()
    with msg.loading(f"Generating train dataset with {train_size} examples..."):
        (train_texts, train_X, train_y) = generate_dataset(train_size, seen_texts)
    msg.loading(f"Train set created with {train_size} examples.")
    with msg.loading(f"Generating eval dataset with {test_size} examples..."):
        (eval_texts, eval_X, eval_y) = generate_dataset(test_size, seen_texts)
    msg.loading(f"Eval set created with {test_size} examples.")
    model = CountLikeTerms(256)
    model.initialize(train_X[:2], train_y[:2])
    optimizer = Adam()

    from thinc.api import fix_random_seed

    fix_random_seed(0)
    for n in range(num_iters):
        loss = 0.0
        batches = model.ops.multibatch(batch_size, train_X, train_y, shuffle=True)
        for X, Y in batches:
            Yh, backprop = model.begin_update(X)
            d_loss = []
            for i in range(len(Yh)):
                d_loss.append(Yh[i] - Y[i])
                loss += ((Yh[i] - Y[i]) ** 2).sum()
            backprop(numpy.array(d_loss))
            model.finish_update(optimizer)
        score = evaluate_model(model, texts=eval_texts, X=eval_X, Y=eval_y)
        print(f"{n}\t{score:.2f}\t{loss:.2f}")

    Yeval = model.predict(eval_X)
    score = evaluate_model(
        model, print_problems=True, texts=eval_texts, X=eval_X, Y=eval_y
    )
    print(f"Score: {score}")

