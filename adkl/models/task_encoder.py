import torch
from os.path import join, dirname, abspath
from torch.nn import Sequential
from torch.nn.functional import mse_loss
from torch.nn.functional import cross_entropy
from torch.utils.data.dataloader import DataLoader
from adkl.feature_extraction import FeaturesExtractorFactory
from adkl.models.base import MetaLearnerRegression, MetaNetwork
from .set_encoding import DatasetEncoderFactory
from .utils import pack_episodes, hash_params, compute_cos_similarity


PRE_TRAINED_ROOT_LOCAL = abspath('../../pretrain_models')
PRE_TRAINED_ROOT_AWS = abspath('pretrain_models')


class TaskEncoderNet(MetaNetwork):

    def __init__(self, input_features_extractor_params,
                 target_features_extractor_params,
                 dataset_encoder_params, complement_module_input_fextractor=None):
        """
        In the constructor we instantiate an lstm module
        """
        super(TaskEncoderNet, self).__init__()
        self.input_fextractor = FeaturesExtractorFactory()(**input_features_extractor_params)
        in_dim = self.input_fextractor.output_dim
        if complement_module_input_fextractor is not None:
            self.complement_i = complement_module_input_fextractor
        else:
            self.complement_i = Sequential()
        self.target_fextractor = FeaturesExtractorFactory()(**target_features_extractor_params)

        out_dim = self.target_fextractor.output_dim
        de_fac = DatasetEncoderFactory()
        self.f_net = de_fac(input_dim=in_dim, output_dim=out_dim, **dataset_encoder_params)
        self.g_net = de_fac(input_dim=in_dim, output_dim=out_dim, **dataset_encoder_params)
        self._output_dim = self.f_net.output_dim
        self.__params = dict(in_params=input_features_extractor_params,
                             out_params=target_features_extractor_params,
                             set_params=dataset_encoder_params)

    @property
    def output_dim(self):
        return self._output_dim

    def get_phis(self, xs, ys, mask):
        xx = self.complement_i(self.input_fextractor(xs.view(-1, xs.shape[2])))
        xx = (xx.t() * mask.view(-1)).t().reshape(*xs.shape[:2], -1)
        yy = self.target_fextractor(ys.view(-1, ys.shape[2]))
        yy = (yy.t() * mask.view(-1)).t().reshape(*ys.shape[:2], -1)
        xxyy = torch.cat((xx, yy), dim=2)
        phis = self.f_net(xxyy)
        return phis

    def forward(self, episodes, xs_train=None, ys_train=None, mask_train=None):

        if episodes is None and xs_train is not None and ys_train is not None:
            return self.get_phis(xs_train, ys_train, mask_train)

        train, test = pack_episodes(episodes, return_ys_test=True)
        xs_train, ys_train, _, mask_train = train
        xs_test, ys_test, _, mask_test = test

        phis_train = self.get_phis(xs_train, ys_train, mask_train)
        phis_test = self.get_phis(xs_test, ys_test, mask_test)

        return compute_cos_similarity(phis_test, phis_train)

    @property
    def weights_filename(self):
        return f'task_encoder_weights_{hash_params(self.__params)}.h5'

    def save(self, directory):
        torch.save(self.state_dict(), join(directory, self.weights_filename))

    @classmethod
    def get_model(cls, *args, pretrained=False, fixe_params=False, **kwargs):
        model = cls(*args, **kwargs)
        if pretrained:
            print('USING PRETRAINED NETWORK')
            try:
                filename = join(PRE_TRAINED_ROOT_LOCAL, model.weights_filename)
                model.load_state_dict(torch.load(filename))
            except FileNotFoundError:
                filename = join(PRE_TRAINED_ROOT_AWS, model.weights_filename)
                model.load_state_dict(torch.load(filename))

        if fixe_params:
            for param in model.parameters():
                param.requires_grad = False
        return model


class TaskEncoderLearner(MetaLearnerRegression):

    def __init__(self, *args, optimizer='adam', lr=0.001, weight_decay=0.0, **kwargs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        network = TaskEncoderNet(*args, **kwargs)
        super(TaskEncoderLearner, self).__init__(network, optimizer, lr, weight_decay)

    def _compute_aux_return_loss(self, y_preds, y_tests):
        y_preds_class = torch.arange(len(y_preds))
        loss = cross_entropy(y_preds, y_preds_class)
        res = dict(loss=loss,
                   accuracy=(y_preds.argmax(dim=1) == y_preds_class).float().mean())
        return loss, res

    def evaluate(self, metatest, metrics=[mse_loss], tboard_folder=None, return_data=False, **kwargs):
        save_dir = abspath(kwargs.get('output_path', '.'))
        self.model.save(save_dir)

        metatest = DataLoader(metatest.dataset, collate_fn=metatest.collate_fn, batch_size=32)
        if hasattr(self, 'model') and hasattr(self, 'writer'):
            self.model.eval()
            if tboard_folder is not None:
                self.writer = SummaryWriter(tboard_folder)

        res_per_task = dict(name=dict(toto='all'), acc=dict(toto=[]))
        data = dict()
        for episodes, _ in metatest:
            y_preds_class = torch.arange(len(episodes))
            y_preds = self.model(episodes)
            accuracy = (y_preds.argmax(dim=1) == y_preds_class).float().mean().data.numpy()
            res_per_task['acc']['toto'].append(accuracy)

        if return_data:
            return res_per_task, data

        return res_per_task


if __name__ == '__main__':
    pass
