from .model import Model


class FeatureExtracter(Model):
    def __init__(self, attrs):
        Model.__init__(self)
        self.attrs = attrs

    def infer_dimensions(self, X=None, Y=None):
        pass

    def begin_update(self, docs):
        # Handle spans
        features = [self._get_feats(doc) for doc in docs]
        return features, _feature_extracter_bwd

    def _get_feats(self, doc):
        if hasattr(doc, "to_array"):
            arr = doc.to_array(self.attrs)
        else:
            arr = doc.doc.to_array(self.attrs)[doc.start : doc.end]
        return self.ops.asarray(arr, dtype="uint64")


def _feature_extracter_bwd(d_features):
    return d_features
