
def main(Xs, labels):
    dXs = zeroslike(inputs)
    dY = zeroslike(labels)

    model = chain(Linear(10), Linear(10))


cdef class Chain:
    def __init__(self, model1, model2):
        self.one = model1
        self.two = model2

    def get_tasks(self, first, third, d_third, d_first, N):
        tasks = []
        self.second = self.mem.alloc(self.one.out_stride, N)
        self.d_second = self.mem.alloc(self.one.out_stride, N)
        tasks.extend(self.one.get_tasks(first, second, d_second, d_first, N))
        tasks.extend(self.two.get_tasks(second, third, d_third, d_second, N))
        return tasks


class Linear(Model):
    def __init__(self, nr_out, nr_in):
        self.nr_out = nr_out
        self.nr_in = nr_in

    def get_tasks(self, inputs, outputs, d_outputs, d_inputs):
        return [Task(self.update, inputs, outputs, d_outputs, d_inputs, N)]


def update_chain(conn, layers, first, third, d_third, d_first): 
    second = conn.create_channel(layer1.output_size)
    d_second = conn.create_channel(layer1.output_size)

    conn.launch_task(layer1.update, (conn, first, second, d_second, d_first))
    conn.launch_task(layer2.update, (conn, second, third, d_third, d_second))
