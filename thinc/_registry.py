import catalogue


class registry(object):
    optimizers = catalogue.create("thinc", "optimizers", entry_points=True)
    schedules = catalogue.create("thinc", "schedules", entry_points=True)
    layers = catalogue.create("thinc", "layers", entry_points=True)

    @classmethod
    def get(cls, name, key):
        if not hasattr(cls, name):
            raise ValueError("Unknown registry: %s" % name)
        reg = getattr(cls, name)
        func = reg.get(key)
        if func is None:
            raise ValueError("Could not find %s in %s" % (name, key))
        return func

    @classmethod
    def make_optimizer(name, args, kwargs):
        func = cls.optimizers.get(name)
        return func(*args, **kwargs)

    @classmethod
    def make_schedule(name, args, kwargs):
        func = cls.schedules.get(name)
        return func(*args, **kwargs)

    @classmethod
    def make_initializer(name, args, kwargs):
        func = cls.initializers.get(name)
        return func(*args, **kwargs)

    @classmethod
    def make_layer(cls, name, args, kwargs):
        func = cls.layers.get(name)
        return func(*args, **kwargs)

    @classmethod
    def make_combinator(cls, name, args, kwargs):
        func = cls.combinators.get(name)
        return func(*args, **kwargs)

    @classmethod
    def make_transform(cls, name, args, kwargs):
        func = cls.transforms.get(name)
        return func(*args, **kwargs)

    @classmethod
    def make_from_config(cls, config, id_start="@"):
        """Unpack a config dictionary, creating objects from the registry 
        recursively.
        """
        id_keys = [key for key in config.keys() if key.startswith(id_start)]
        if len(id_keys) >= 2:
            raise ValueError("Multiple registry keys in config: %s" % id_keys)
        elif len(id_keys) == 0:
            # Recurse over subdictionaries, filling in values.
            filled = {}
            for key, value in config.items():
                if isinstance(value, dict):
                    filled[key] = cls.make_from_config(value, id_start=id_start)
                else:
                    filled[key] = value
            return filled
        else:
            getter = cls.get(id_keys[0].replace(id_start, ""), config[id_keys[0]])
            args = []
            kwargs = {}
            for key, value in config.items():
                if isinstance(value, dict):
                    value = cls.make_from_config(value, id_start=id_start)
                if isinstance(key, int) or key.isdigit():
                    args.append((int(key), value))
                elif not key.startswith(id_start):
                    kwargs[key] = value
            args = [value for key, value in sorted(args)]
            return getter(*args, **kwargs)
