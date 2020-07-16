import ray
from spacy.cli.train import load_nlp_and_config, msg, train_while_improving
from spacy.cli.train import create_train_batches, create_evaluation_callback
from spacy.cli.train import setup_printer
from spacy.gold import Corpus
from thinc.backends._ray_remote_params import RayProxy, SharedOptimizer


class Worker:
    def __init__(self, rank, num_workers, paths, use_gpu):
        self.rank = rank
        self.num_workers = num_workers
        self.paths = paths
        self.gpu_id = self._resolve_gpu(use_gpu)
        self.nlp, self.config = load_nlp_and_config(paths["config"])
        self.corpus = Corpus(
            paths["train"],
            paths["dev"],
            limit=self.config["training"]["limit"]
        )
        self._initialize_models(self.nlp, self.corpus, self.config)
        self._evaluation_callback = None
        self._results = []

    def get_optimizer(self):
        return self.config["training"]["optimizer"]
 
    def train(self, use_gpu, conn, evaluater=None):
        def evaluate():
            if self.rank == 0:
                return ray.get(evaluater.evaluate.remote())
            else:
                return ray.get(evaluater.get_last_evaluation.remote())

        self.config["training"]["batch_size"] //= self.num_workers
        self._set_params_proxies(self.nlp, conn)
        train_batches = create_train_batches(
            self.nlp,
            self.corpus,
            self.config["training"],
            self.rank
        )

        training_step_iterator = train_while_improving(
            self.nlp,
            FakeOptimizer(conn, self.rank),
            train_batches,
            evaluate=evaluate,
            dropout=self.config["training"]["dropout"],
            accumulate_gradient=self.config["training"]["accumulate_gradient"],
            patience=self.config["training"].get("patience", 0),
            max_steps=self.config["training"].get("max_steps", 0),
            eval_frequency=self.config["training"]["eval_frequency"],
            raw_text=None,
        )
        if self.rank == 0:
            print_row = setup_printer(self.config["training"], self.nlp.pipe_names)
        output_path = self.paths.get("output_path")
        for batch, info, is_best_checkpoint in training_step_iterator:
            if self.rank == 0 and is_best_checkpoint is not None:
                info["words"] *= self.num_workers
                print_row(info)
                if is_best_checkpoint and output_path is not None:
                    self.save_checkpoint(info, output_path / "model-best")

    def evaluate(self):
        if self._evaluation_callback is None:
            self._evaluation_callback = create_evaluation_callback(
                self.nlp,
                self.config["training"]["optimizer"],
                self.corpus,
                self.config["training"]
            )
        self._results.append(self._evaluation_callback())
        return self._results[-1]

    def save_checkpoint(self, info, output_path):
        update_meta(self.config["training"], self.nlp, info)
        self.nlp.to_disk(output_path)

    def get_last_evaluation(self):
        if not self._results:
            return self.evaluate()
        else:
            return self._results[-1]

    def _resolve_gpu(self, use_gpu):
        if use_gpu >= 0:
            gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES")
            msg.info(f"Using GPU (isolated): {gpu_id}")
            require_gpu(0)
        else:
            msg.info("Using CPU")
            gpu_id = -1
        return gpu_id

    def _initialize_models(self, nlp, corpus, config):
        train_examples = list(
            corpus.train_dataset(
                nlp,
                shuffle=False,
                gold_preproc=config["training"]["gold_preproc"],
                max_length=config["training"]["max_length"],
            )
        )
        nlp.begin_training(lambda: train_examples)

    def _set_params_proxies(self, nlp, conn):
        proxy = RayProxy(conn)
        for name, component in nlp.pipeline:
            if hasattr(component, "model"):
                component.model.set_params_proxy(proxy)


class FakeOptimizer:
    def __init__(self, conn, worker_id):
        self.conn = conn
        self.worker_id = worker_id
        self._futures = []

    def __call__(self, key, weights, gradient):
        raise ValueError("Should not be called?")

    def step_schedules(self):
        ray.get(self._futures)
        self._futures = []
        if self.worker_id == 0:
            self._futures.append(self.conn.step_schedules.remote())
        self._futures.append(self.conn.inc_progress.remote(self.worker_id))


def distributed_setup_and_train(
    use_gpu,
    num_workers,
    strategy,
    ray_address,
    paths,
    quorum=None
):
    print("Use ray", num_workers)
    if quorum is None:
        quorum = num_workers
    if ray_address is not None:
        ray.init(address=ray_address)
    else:
        ray.init(ignore_reinit_error=True)

    RemoteWorker = ray.remote(Worker).options(
        num_gpus=int(use_gpu >= 0),
        num_cpus=1
    )
    workers = [
        RemoteWorker.remote(rank, num_workers, paths, use_gpu=use_gpu)
        for rank in range(num_workers)
    ]
    evaluater = RemoteWorker.remote(num_workers + 1, num_workers, paths, use_gpu=use_gpu)
    optimizer = workers[0].get_optimizer.remote()

    conn = ray.remote(SharedOptimizer).remote(quorum, optimizer)
    futures = [] 
    for i, w in enumerate(workers):
        futures.append(w.train.remote(use_gpu, conn, evaluater))
    ray.get(futures)
