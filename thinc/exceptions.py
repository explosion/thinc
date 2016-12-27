class ShapeError(ValueError):
    @classmethod
    def expected_batch(cls, locals_, globals_):
        return cls("Expected batch")
    
    @classmethod
    def dim_mismatch(cls, dims, locals_, globals_):
        return cls("Dimension mismatch: %s vs %s" % dims)
