from .maml import MAML
from .mann import MANN
from .snail import SNAIL
from .metakrr_sk import MetaKrrSKLearner
from .metakrr_mk import MetaKrrMKLearner
from .metagp_mk import MetaGPMKLearner
from .fp_learner import FPLearner
from .cnp import CNPLearner
from .irl_matching_nets import IrlMachtingNetLearner
from .multitask import MultiTaskLearner
from .task_encoder import TaskEncoderLearner
from .light_bmaml import LightBMAML
from .proto_maml import ProtoMAML
from .learned_basis import LearnedBasisFunctions
from .r2d2 import R2D2Learner
from .single_task_learner import STLearner
from .metakrr_mk2 import MetaKrrMKLearner2, MetaGPMKLearner2


class ModelFactory:
    name_map = dict(
        mann=MANN,
        maml=MAML,
        snail=SNAIL,
        metakrr_sk=MetaKrrSKLearner,
        metakrr_mk=MetaKrrMKLearner,
        metakrr_mk2=MetaKrrMKLearner2,
        metagp_mk=MetaGPMKLearner,
        metagp_mk2=MetaGPMKLearner2,
        fingerprint=FPLearner,
        cnp=CNPLearner,
        irl=IrlMachtingNetLearner,
        multitask=MultiTaskLearner,
        task_encoder=TaskEncoderLearner,
        bmaml=LightBMAML,
        protomaml=ProtoMAML,
        learned_basis=LearnedBasisFunctions,
        r2d2=R2D2Learner,
        single_task=STLearner
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
