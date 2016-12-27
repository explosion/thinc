import thinc


def train(device, train_data, save_to):
    Model = thinc.recommend(train_data)
    model = Model(device, name='my_model')
    with model.begin_training(train_data) as trainer, optimizer:
        for inputs, truth in trainer.iterate(train_data):
            guess, finish_update = model.begin_update(inputs)
            finish_update(truth - guess, optimizer)
    model.pip_export(save_to)


if __name__ == '__main__':
    thinc.cli(train)
