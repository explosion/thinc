from .types import Decorator

import catalogue


class registry(catalogue.catalogue_registry):
    # fmt: off
    optimizers: Decorator = catalogue.create("thinc", "optimizers", entry_points=True)
    schedules: Decorator = catalogue.create("thinc", "schedules", entry_points=True)
    layers: Decorator = catalogue.create("thinc", "layers", entry_points=True)
    losses: Decorator = catalogue.create("thinc", "losses", entry_points=True)
    initializers: Decorator = catalogue.create("thinc", "initializers", entry_points=True)
    datasets: Decorator = catalogue.create("thinc", "datasets", entry_points=True)
    ops: Decorator = catalogue.create("thinc", "ops", entry_points=True)
    # fmt: on


__all__ = ["registry"]
