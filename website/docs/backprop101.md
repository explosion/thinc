---
title: Backpropagation 101
teaser: How to trick yourself into understanding backprop without even trying
---

Imagine you're a project manager, somewhere deep inside a vast company. You have
an inbox, an outbox, and three people in your team: Alex, Bo, and Casey. Work
comes into your inbox, you allocate it to someone in your team, they perform
the work and get the results back to you, and you move those results to your
outbox. Some time later, and potentially out-of-order, you'll receive feedback
on the work you submitted. Of course, when you receive the feedback, it won't be
labelled according to the person who did it --- the bureaucracy above you
neither knows nor cares about Alex, Bo and Casey. All the tasks will have an ID
attached, and you'll pass that ID on when you move the results forward. Later
you'll use the ID to figure out how to handle the feedback.

```python
def handle_work(team, inbox, outbox, feedback_from_above):
    ...
    while True:
        if not_empty(inbox):
            task_id, task = next(inbox)
            worker = choose_worker(team)
            results = worker(task)
            outbox.send(task_id, results)
        if not_empty(feedback_from_above):
            task_id, feedback = next(feedback_from_above)
            ...
```

Because you don't know when you'll get the feedback, you have some state to
track. You need to keep track of who did what task, so you can route the
feedback to the right person. If Alex did the task and you give the feedback to
Bo, your team will not improve. And your team members have some state to track
too: they need to understand each piece of feedback in terms of the specific
task it relates to. You want them to be able to get feedback like, "This wasn't
ambitious enough", notice that their proposal was way under-budget, and see what
they should've done differently in that specific scenario.

Alex, Bo and Casey should each keep their own notes about their projects, and
the specifics of exactly what they should do differently should be up to them
--- you don't want to micromanage that. You have a great team. If you just route
the information around, they'll take care of the rest. So to make your routing
job easier, you ask everyone to return you a _callback_ to pass along the
feedback when it's ready. The callback is created by the worker on your team,
and it should wrap whatever state they need to act upon the feedback when/if it
comes. With this system, all you need to do is file all the callbacks correctly
when you're passing work forward, and then retrieve the right handle when the
feedback comes in.

```python
def handle_work(team, inbox, outbox, feedback_from_above):
    pending_feedback = {}
    while True:
        if not_empty(inbox):
            task_id, task = next(inbox)
            worker = choose_worker(team)
            results, handle_feedback = worker(task)
            pending_feedback[task_id] = handle_feedback
            outbox.send(task_id, results)
        if not_empty(feedback_from_above):
            task_id, feedback = next(feedback_from_above)
            handle_feedback = pending_feedback[task_id]
            handle_feedback(feedback)
```

This system definitely makes your job easy, and all the information is getting
routed correctly. But something's still missing. Alex, Bo and Casey have
feedback too: about their inputs. They are getting feedback about their work,
and are doing their best to incorporate it and improve. But their own
performance is also dependent on the specific inputs they were given. It's
always the case that if the input for a given task had been a little bit
different, the output would've been different as well. When incorporating the
feedback on their work, the workers in your team will thus have feedback on the
original inputs they were given, and how those inputs could have been better to
ensure outputs that would have been closer to what the people above you really
wanted. Currently all this feedback is getting lost, and there's no way for the
people who produced those inputs to learn what your team wants from them. So you
need another outbox, pointed in the other direction, to propagate the feedback
from your workers to the people preparing their inputs.

```python
def handle_work(team, inbox, outbox, feedback_from_above, feedback_to_below):
    ...
```

Of course, you need to make a clear distinction between the feedback that your
team received on their outputs, and the feedback that your team produced about
their inputs, and make sure that the correct pieces of feedback end up with the
right people.

Imagine if Alex had created a proposal that could potentially run over-budget,
and you had passed that proposal upwards. Later you pass along feedback to Alex
that says: "Not ambitious enough; client asked for bold". That's a feedback
message for Alex, about Alex's work. The team who made the input proposal then
needs to hear Alex's feedback on the original inputs: "The client context was
originally described as 'risk sanctioned', which is ambiguous phrasing. Please
be more clear when specifying the requirements." If instead you passed them the
feedback intended for Alex, the team below you would be misled. They'd move in
the wrong direction. So you need to be careful that everything's routed
correctly. The feedback into Alex and the feedback out of Alex are not
interchangeable.

```python
def handle_work(team, inbox, outbox, feedback_from_above, feedback_to_below):
    pending_feedback = {}
    while True:
        if not_empty(inbox):
            task_id, task = next(inbox)
            worker = choose_worker(team)
            results, handle_feedback = worker(task)
            pending_feedback[task_id] = handle_feedback
            outbox.send(task_id, results)
        if not_empty(feedback_from_above):
            task_id, feedback = next(feedback_from_above)
            handle_feedback = pending_feedback[task_id]
            feedback_to_below.send(task_id, handle_feedback(feedback))
```

With work passing forward, and corrections being fed back, your team and the
people feeding you work are operating smoothly. The corrections you're asking
Alex, Bo and Casey to make get more and more minor; and in turn, the corrections
they're passing back are getting smaller too. Life is good, work is easy... So
you start to have some time on your hands.

One day you're watching a TED talk about management, and you hear about the
"wisdom of crowds": if you combine several independent estimates, you can get a
more accurate result. You only have a crowd of three, but you're getting a lot
of budget estimation tasks, so why not give it a try?

For the next budget estimation task, instead of giving it to just one worker,
you decide to get them all to work on it separately. You don't tell them about
each others' work, because you don't want groupthink --- you want them to all
come up with a separate estimate. You then add them all up, and send off the
result.

```python
alex_estimate, give_alex_feedback = alex(task)
bo_estimate, give_bo_feedback = bo(task)
casey_estimate, give_casey_feedback = casey(task)
estimate = alex_estimate + bo_estimate + casey_estimate
```

Looking back on what you know now, just adding them up does feel kind of
silly... But the rest of the TED talk was some weird stuff about jellybeans and
you stopped paying attention. So this is what you did. Anyway, the estimate you
sent off was way too high, so now you'd better give everyone the feedback so
they can adjust for next time. Since this was a numerical estimate, the feedback
is very simple: it's just a number. You don't actually know very much about this
number and what it really represents, but you get a bonus if your team outputs
work such that smaller numbers come back. The closer to 0 the feedback becomes,
the bigger the bonus. You incentivise your team accordingly.

The first time you just sum up their estimates, the combined estimate way
overshoots, and your feedback is far from zero. How should you split up the
feedback between Alex, Bo and Casey, and when they have feedback in turn, how
should you pass that along?

```python

def propagate_feedback_from_addition(feedback, give_alex_feedback, give_bo_feedback, give_casey_feedback):
    # What to do here?
    ...
    return feedback_to_input
```

One way to think about this is that there's three people, and one piece of
feedback. So we should divide it up between all three estimates equally. But you
think about this some more, and decide that you really don't feel like
micromanaging this: you just want the _combined_ score to come out right. So you
figure that you'll just give all of the feedback to everyone, and see how that
works out. Sure, your team may be a bit confused at first, but they'll quickly
adjust their spreadsheets or whatever and the combined estimate will get closer
and closer to the mark.

There's also the question of how to pass on Alex, Bo and Casey's feedback about
their inputs. It turns out for these cost estimates, everything comes in a
nicely structured format: the "inputs" are just a table of numbers, which were
all estimates from another team, who are passing information into your inbox. So
Alex, Bo and Casey all produce feedback that's in the same format --- it's a
table of numbers of the same size and shape as the inputs (because that's what
it relates to).

Alex, Bo and Casey take their bonuses seriously, so they're very specific about
how their feedback should be interpreted. Each of them gives you their feedback
table and tells you, "Look, tell the people producing this data that if they had
given us inputs such that these numbers were zero, our own feedback would have
been zero and we'd all make our bonus. Now, I know we shouldn't leap to
conclusions and base everything off this one sample. And I know I need to make
adjustments as well. If I make some changes and they make some changes each
time, we'll get there after a bit of iteration."

So now you have three of these feedback tables, that all relate to the same
example. How should you propagate that back? The only sensible thing is to
add them all up and pass them on, so that's what you do. More of these work
estimates come in, and you keep passing them through your combination team and
passing the feedback back down. It's kind of a hassle to keep track of all the
internals though --- it's messed up your neat system. So you have a bright idea:
you create a little filing system for yourself, so you can keep track of the
combination and treat it just like another team member.

Alex, Bo and Casey all behave with a pretty simple interface when it comes to
these estimates, because the data is all nice and regular. We can specify the
interface using Python 3's type-annotation syntax, so we can understand what
data we're passing around a bit better. The inputs are a table, so we'll write
their type as `Array2d` --- i.e., a two-dimensional array. The output will be a
single number, so a float. Each worker also returns a callback, to handle the
feedback about their output, and provide the feedback about their inputs.

```python
def estimate_project(inputs: Array2d) -> Tuple[float, Callable[[float], Array2d]]:
    ...
```

It'll be helpful if we can refer to objects that follow this `estimate_projects`
API in the type annotations. The type-annotations solution to this is to define
a "protocol". The specifics are a bit weird, but it comes out looking like this:

```python
from typing import Protocol

class Estimator(Protocol):
    def __call__(self, inputs: Array2d) -> Tuple[float, Callable[[float], Array2d]]):
        ...
```

This gives us a new type, `Estimator`, that we can use to describe our worker
functions. As we start combining workers, we'll be passing functions into
functions --- so it'll be helpful to have some annotations to see what's going
on more easily.

To make our combination worker, we just need to return a function that has the
same signature. Inside the addition estimator, we'll call Alex, Bo and Casey in
turn, add up the output, and return it along with the callback. For notational
convenience, we'll prefix the feedback for some quantity with `re_`, like it's a
reply to that variable.

```python

def combine_by_addition(alex: Estimator, bo: Estimator, casey: Estimator) -> Estimator:

    def run_addition_estimate(inputs: Array2d) -> float:
        a_estimate, give_a_feedback = alex(inputs)
        b_estimate, give_b_feedback = bo(inputs)
        c_estimate, give_c_feedback = casey(inputs)

        summed = a_estimate + b_estimate + c_estimate

        def handle_feedback(re_summed: float) -> Array2d:
            # Pass the feedback re output 'summed' to each worker, and add up their
            # feedbacks re input
            re_input = (
                give_a_feedback(re_summed)
                + give_b_feedback(re_summed)
                + give_c_feedback(re_summed)
            )
            return re_input

        return summed, handle_feedback

    return run_addition_estimate
```

We can now use our "addition" worker just like anyone else in our team. And in
fact, if we learned tomorrow that "Casey" was actually a front for a vast system
of combination like this... well, what of it? We'd still be passing in inputs,
passing along the outputs, providing the output feedback, and making sure the
input feedback gets propagated. Nothing would change.

After a few iterations of corrections, the combined-by-addition "worker" you've
created starts producing great results --- so good that even the vast
bureaucracy around you takes notice. As well as a great bonus, you get a few new
team members: Dani, Ely and Fei. You start thinking of new ways to combine them.
You also make some quick changes to your addition system. Now that you have more
workers, you want to make it a bit more general.

```python

def combine_by_addition(workers: List[Estimator]) -> Estimator:

    def addition_combination(inputs: Array2d) -> float:
        callbacks = []
        summed = 0
        for worker in workers:
            result, callback = worker(inputs)
            summed += result
            callbacks.append(worker)

        def handle_feedback(re_summed: float) -> Array2d:
            re_input = callbacks[0](re_summed)
            for callback in callbacks[1:]:
                re_input += callback(re_summed)
            return re_input

        return summed, handle_feedback

    return addition_combination
```

As for new combinations, one obvious idea harks back to your original "wisdom of
the crowds" inspiration. Instead of just adding up the outputs, you could
average them. Easy. But how to handle the feedback? Should we just pass that
along directly, like we did with the addition, or should we divide the feedback
by the number of workers?

It actually won't really matter: the team members all understand the feedback to
mean, "Change your model slightly, so that this number becomes closer to zero.
Also, give us similar feedback about inputs." If you give them feedback that's
three times too big, and they make changes that pushes that number towards zero,
they'll also be pushing the "real" feedback score towards zero. You can't really
steer them wrong just by messing up the magnitude, so long as you do it
consistently. Still, messing up the magnitude makes things messy: if you're not
careful, it could easily lead to more relevant errors later. So best to handle
everything consistently, and make the appropriate division.

```python

def combine_by_average(workers: List[Estimator]) -> Estimator:

    def combination_worker_averaging(inputs: Array2d) -> float:
        callbacks = []
        summed = 0
        for worker in workers:
            result, callback = worker(inputs)
            summed += result
            callbacks.append(worker)
        average = summed / len(workers)

        def handle_feedback(re_average: float) -> Array2d:
            re_result = re_average / len(workers)
            re_input = callbacks[0](re_result)
            for callback in callbacks[1:]:
                re_input += callback(re_result)
            return re_input

        return average, handle_feedback

    return combination_worker_averaging
```

Looking at this, there's a lot of obvious duplication with the addition. We're
doing the exact same thing as it, as part of the averaging process. Why don't we
just make an addition worker, and only implement the averaging step?

```python

def combine_by_average(workers: List[Estimator]) -> Estimator:
    addition_worker = combine_by_addition(workers)

    def combination_worker_averaging(inputs: Array2d) -> float:
        summed, handle_summed_feedback = addition_worker(inputs)
        average = summed / len(workers)

        def handle_feedback(re_average: float) -> Array2d:
            re_summed = re_average / len(workers)
            re_input = handle_summed_feedback(re_summed)
            return re_input

        return average, handle_feedback

    return combination_worker_averaging
```

If you only use each worker in one team, and you keep the team sizes fixed, the
addition and averaging approaches end up performing the same. The extra division
step doesn't end up mattering. This actually makes a lot of sense, considering
what we realized about the feedback for the averaging: in both approaches, the
workers are going to end up making similar updates, just rescaled --- and over
time, they'll easily recalibrate their outputs to the scaling term, either way.

Summing and averaging are sort of the same, but surely there are other ways the
workers could collaborate? So you go back and read more management books, and
everyone seems to be saying you should just listen to whoever speaks the
loudest. None of the books say it _like that_, but if you actually followed
their advice, that's pretty much what you'd end up doing. This seems really
dumb, but uh... okay, let's try it? Your team really doesn't communicate by
speaking, so "loudest" can't be taken too literally. Let's take it to mean
selecting the highest estimate.

```python

def combine_by_maximum(workers: List[Estimator]) -> Estimator:

    def combination_worker_maximum(inputs: Array2d) -> float:
        max_estimate = None
        handle_for_max = None
        for worker in workers:
            estimate, handle_feedback = worker(inputs)
            if max_estimate is None or estimate > max_estimate:
                max_estimate = estimate
                handle_for_max = handle_feedback
        return max_estimate, handle_for_max

    return combination_worker_maximum
```

You combine two workers, `Dani` and `Ely`, into a new team using this
maximum-based approach, and you can almost feel your bonus slipping away as you
put them into action: surely if we're always taking the maximum, our estimates
are going to climb up and up, right? But to your surprise, that's not what
happens. The worker who submits the high estimate is the one who gets the
feedback, so they'll learn not to produce such a high estimate for that input
next time. Dani and Ely aren't competing to have their outputs selected --- from
their perspective, they're working completely independently. They're just trying
to make adjustments so that their feedback scores get closer to zero.

Is it weird that only the worker with the highest estimate gets any feedback?
Shouldn't we be trying to train all of them based on what we learned from the
example as well? We actually can't do that, because we don't have feedback that
relates to all the outputs: we only submitted one output, so we only get
feedback about that one output. The feedback represents a request for change: it
tells the workers how we'd like their output to be different next time, given
the same input. We don't know that about the other workers' estimates, because
we didn't submit them.

Your `combine_by_maximum(Dani, Ely)` team works surprisingly well, so you decide
to break your usual hands-off policy, and actually look at some of the data to
try to figure out what's going on, even going so far as to set up a
`combine_by_average(Alex, Bo)` team for comparison. After a bit of sifting, you
discover some interesting patterns, especially concerning two of the input
columns.

Based on the estimates and feedback, you see that if the inputs have a 1 in the
column labelled "Located in California", that generally means the estimates
should be higher. There's also a column labelled "Renewable Energy", and a 1 for
that also leads to higher estimates, generally. But when there's a 1 for both
columns, the estimates should come out a fair bit lower than you'd expect, based
on the two columns individually.

The `combine_by_average(Alex, Bo)` team is really struggling with this: whatever
they're doing individually, it's not taking this combination factor into account
--- they're both overshooting on the `California+Renewable` examples, and when
there's a run of those examples, they start _undershooting_ on the examples that
are just `California` or just `Renewable`. The average doesn't help.

| California | Renewables | Alex    | Bo     | Output | Target |
| ---------- | ---------- | ------- | ------ | ------ | ------ |
| 0          | 0          | \$1.5m  | \$0.5m | \$1m   | \$1m   |
| 1          | 0          | \$6m    | \$4m   | \$5m   | \$5m   |
| 0          | 1          | \$7m    | \$9m   | \$8m   | \$8m   |
| 1          | 1          | \$11.5m | \$12.5 | \$12m  | \$10m  |

While the averaging doesn't help, the `combine_by_maximum(Dani, Ely)` team
manages to follow their individual feedbacks to an interesting "collaborative"
solution. Effectively, the `max` operation allows the workers to "specialise":
Dani doesn't worry about examples outside of California, and Ely doesn't worry
about projects that don't concern renewables. This means Ely's weighting for
"California" is really a weighting for _California in the context of renewable_.
Ely doesn't need to model the effect of California by itself, because Dani
covers that, so between them, they're able to produce estimates that account for
the interaction effect.

| California | Renewables | Dani | Ely   | Output | Target |
| ---------- | ---------- | ---- | ----- | ------ | ------ |
| 0          | 0          | \$1m | \$1m  | \$1m   | \$1m   |
| 1          | 0          | \$5m | \$2m  | \$5m   | \$5m   |
| 0          | 1          | \$1m | \$8m  | \$8m   | \$8m   |
| 1          | 1          | \$6m | \$10m | \$10m  | \$10m  |

It's hard to overstate the importance of this. Nobody involved went out and
learned that there were relevant subsidies in California that reduced the cost
of renewables projects, figured out that they were throwing off the estimates,
and included an extra column for them. And later more examples will show even
more subtlety: the subsidies only matter for certain years. That fact gets taken
care of too, without anyone even having to notice.

The `combine_by_maximum` approach can learn "not and" relationships, which is
something summing and averaging the different outputs could never give you. Once
you start looking for places where summing and averaging fails, you start seeing
them everywhere. It will make a great TED talk some day: _non-linearities_.

Previously when you thought about relationships between quantities, you were
trying to decide between three options: unrelated, positively correlated and
negatively correlated. This often comes down to "is this factor good, bad, or
irrelevant". How much salt should someone have in their diet per day? How much
sleep should you get? How long should a school day be?

There are very few relationships that are linear the whole way through. It's
much more common for the relationship to be linear in "normal" ranges, and then
non-linear at the edges. Often there's a saturation point: you need enough
sodium in your diet. If you have too little you will die. But once you have
enough, excess sodium is probably slightly bad for you, up until a point where
it will be extremely bad for you, and again, you will die. Almost everything is
like this, because almost every relationship is indirect --- we're always
looking at aggregate phenomena with a huge web of causation behind the scenes,
mediating the interaction. So you'll always have these pockets of linearity,
interrupted by overheads, tipping points, cut-offs at zero, saturation points,
diminishing returns, synergistic amplifications, etc.

So the ability to model these non-linear relationships is no small thing. And
it's happening through this simple process of feedback and adjustment: with
enough examples, the individual predictors are getting more right over time, and
you're able to combine their predictions together to account for "not and"
relationships between logical variables, and to interpret numeric variables as
having different significance in different ranges. Business is good, the bonuses
keep flowing, and your team expands further.

As you succeed further, efficiency starts to become a factor. Instead of
receiving one task at a time, you start processing the work in batches. This is
helpful because there's a bit of routing overhead involved. There are also
little waiting periods after they finish their work, as work is happening in
parallel, and sometimes you need to wait on another input somewhere else before
the next bit of work can get started. Batch processing keeps everyone busier,
but it does make your routing a bit more difficult sometimes.

For some of the tasks you're given, the workers will take in a whole table of
numbers and give you a single number back. For other tasks, you get one number
per row. For this second type of task, you think you see a useful way to do the
batching --- but you want to do a quick experiment first. You need to know
whether the order of the rows matter or are they really independent?

```python

def with_shuffled(worker):

    def worker_with_shuffled(input_table):
        shuf_index = list(range(len(input_table)))
        random.shuffle(shuf_index)
        shuf_input = [input_table[i] for i in shuf_index]
        shuf_output, handle_shuffled_feedback = worker(shuf_input)
        # We should undo our mischief before we return the output -- we don't
        # know who else might be relying on the original order.
        # We swapped items at pairs (0, shuf_index[0]), etc --
        # So we can unswap.
        shuf_index_reverted = [shuf_index.index(i) for i in list(range(len(input_table)))]
        output = [shuf_output[i] for i in shuf_index_reverted]

        def handle_feedback(re_output):
            # Our worker received the rows in a different order. We need to
            # align the feedback to the view of the data they received.
            shuf_re_output = [re_output[i] for i in shuf_index]
            shuf_re_input = handle_shuffled_feedback(shuf_re_output)
            # And unshuffle, to return.
            re_input = [shuf_re_input[i] for i in shuf_index_reverted]
            return re_input

        return output, handle_feedback()

    return worker_with_shuffled
```

You decide to grab Alex for this test, and do something just like the
"combination worker" you created earlier, but this time with just one worker.
You quickly determine that your trickery is making no difference: the order of
rows doesn't matter in this input. So now you can handle the batched data
easily. You just need to concatenate the rows to form one giant table, submit it
to the worker, and split apart the results. Then you need to do the inverse with
the feedback. Give it a try!

```python

def with_flatten(worker):
    def worker_with_flatten(input_tables: List[Array2d]) -> Tuple[List[Array1d], Callable]:
        ...

        def handle_flattened_feedback(re_outputs: List[Array1d]) -> List[Array2d]:
            ...
            return re_input_tables

        return outputs, handle_flattened_feedback

    return worker_with_flatten
```

By now it's probably worth dropping the allegory: the "workers" in our story
are models, which could be individual layers of a neural network, or even whole
models. And the process we've been discussing is of course the backpropagation
of gradients, which are used to iteratively update the weights of a model.

The allegory also introduced Thinc's particular implementation strategy for
backpropagation, which uses function composition. This approach lets you
express neural network operations as higher-order functions. On the one hand,
there are sometimes where managing the backward pass explicitly is tricky, and
it's another place your code can go wrong. But the trade-off is that there's
much less API surface to work with, and you can spend more time thinking about
the computations that should be executed, instead of the framework that's
executing them. For more about how Thinc is put together, read on to its
[Concept and Design](/docs/concept).
