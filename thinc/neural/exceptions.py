class ShapeError(ValueError):
    @classmethod
    def expected_batch(cls, locals_, globals_):
        return cls("Expected batch")
    
    @classmethod
    def dim_mismatch(cls, expected, observed):
        return cls("Dimension mismatch: %s vs %s" % (expected, observed))
