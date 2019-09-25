from .cnn import ClonableModule, Cnn1dFeaturesExtractor
from .fc import FcFeaturesExtractor
from .gin import GINFeaturesExtractor
from .transformers import AdjGraphTransformer, SequenceTransformer, FingerprintsTransformer
from .utils import ClonableModule


class FeaturesExtractorFactory:
    name_map = dict(
        fc=FcFeaturesExtractor)

    def __init__(self):
        super(FeaturesExtractorFactory, self).__init__()

    def __call__(self, arch, **kwargs):
        if arch not in self.name_map:
            raise Exception(f"Unhandled feature extractor. The name of \
             the architecture should be one of those: {list(self.name_map.keys())}")
        fe_class = self.name_map[arch.lower()]
        feature_extractor = fe_class(**kwargs)
        return feature_extractor

# if __name__ == "__main__":
#     factory = FeaturesExtractorFactory()
#     factory(arch='fc', input_size=100, hidden_sizes=200, normalize_features=True)
