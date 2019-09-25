from .maml import MAML
from .cnp import CNPLearner
from .light_bmaml import LightBMAML
from .proto_maml import ProtoMAML
from .learned_basis import LearnedBasisFunctions
from .r2d2 import R2D2Learner
from .metakrr_mk2 import MetaKrrMKLearner2, MetaGPMKLearner2


class ModelFactory:
    name_map = dict(
        maml=MAML,
        metakrr_mk2=MetaKrrMKLearner2,
        metagp_mk2=MetaGPMKLearner2,
        cnp=CNPLearner,
        bmaml=LightBMAML,
        protomaml=ProtoMAML,
        learned_basis=LearnedBasisFunctions,
        r2d2=R2D2Learner
    )

    def __init__(self):
        super(ModelFactory, self).__init__()

    def __call__(self, model_name, **kwargs):
        if model_name not in self.name_map:
            raise Exception(f'Unhandled model "{model_name}". The name of '
                            f'the model should be one of those: {list(self.name_map.keys())}')
        modelclass = self.name_map[model_name.lower()]
        model = modelclass(**kwargs)
        return model


if __name__ == "__main__":
    factory = ModelFactory()
    model = factory(arch='fc', input_size=100, hidden_sizes=200, normalize_features=True)
