class ShapeError(ValueError): # pragma: no cover
    @classmethod
    def expected_batch(cls):
        return cls("Expected batch")
    
    @classmethod
    def dim_mismatch(cls, expected, observed):
        return cls("Dimension mismatch: %s vs %s" % (expected, observed))
