import six
import torch
from adkl.models.utils import to_unit, PlainWriter, pack_episodes
from adkl.utils.callbacks import MseMetaTest
from poutyne.framework import warning_settings, Model
from poutyne.framework.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, \
    BestModelRestore, Callback
from tensorboardX import SummaryWriter
from torch.nn.functional import mse_loss

warning_settings['batch_size'] = 'ignore'


def get_optimizer(optimizer, *args, **kwargs):
    r"""
    Get an optimizer by name. cUstom optimizer, need to be subclasses of :class:`torch.optim.Optimizer`.

    Arguments
    ----------
        optimizer: :class:`torch.optim.Optimizer` or str
            A class (not an object) or a valid pytorch Optimizer name

    Returns
    -------
        optm `torch.optim.Optimizer`
            Class that should be initialized to get an optimizer.s
    """
    OPTIMIZERS = {k.lower(): v for k, v in vars(torch.optim).items()
                  if not k.startswith('__')}
    if not isinstance(optimizer, six.string_types) and issubclass(optimizer.__class__, torch.optim.Optimizer):
        return optimizer
    return OPTIMIZERS[optimizer.lower()](*args, **kwargs)


class MetaNetwork(torch.nn.Module):
    @property
    def return_var(self):
        return False

    def __forward(self, episode):
        raise NotImplementedError()

    def forward(self, episodes):
        return [self.__forward(episode) for episode in episodes]

    def forward_active(self, episodes, budget, return_all_steps=False, no_strategy=False):
        if not self.return_var:
            raise Exception("This model can not be use for active learning")

        train, test, query = pack_episodes(episodes, return_ys_test=True, return_query=True)
        ys_test, lens_test = test[1], test[2]
        test = (test[0],) + test[2:]
        n_batch = train[0].shape[0]

        all_test_evals = []

        for i in range(budget + 1):
            # train and predict on preds and test
            xs_train, ys_train, lens_train, masks_train = train
            xs_query, ys_query, lens_query, masks_query = query
            res = self.forward(None, train, test, query, trim_ends=False)
            (y_mean_test, y_std_test), (y_mean_query, y_std_query) = res

            # acquisition step
            if no_strategy:
                new_query_idx = torch.randperm(xs_query.shape[1])[:n_batch]
            else:
                q_vals = y_std_query.squeeze(dim=2)
                q_vals = (q_vals * masks_query)
                new_query_idx = torch.argmax(q_vals, dim=1)

            # build new training set
            rangeb = torch.arange(n_batch)
            temp = xs_query[rangeb, new_query_idx].unsqueeze(1)
            xs_train = torch.cat((xs_train, temp), dim=1)
            temp = ys_query[rangeb, new_query_idx].unsqueeze(1)
            ys_train = torch.cat((ys_train, temp), dim=1)
            masks_train = torch.cat((masks_train, torch.ones_like(masks_train)[:, :1]), dim=1)
            lens_train += 1
            train = (xs_train, ys_train, lens_train, masks_train)

            # build new query setn
            masks_query[rangeb, new_query_idx] = 0
            lens_train -= 1
            query = (xs_query, ys_query, lens_query, masks_query)

            # save results
            all_test_evals.append([(mu[:n], std[:n]) for n, mu, std in zip(lens_test, y_mean_test, y_std_test)])

        if return_all_steps:
            return all_test_evals
        else:
            return all_test_evals[-1]


class MetaLearnerRegression(Model):
    def __init__(self, network, optimizer, lr, weight_decay):
        if torch.cuda.is_available():
            network.cuda()
        opt = get_optimizer(optimizer, filter(lambda p: p.requires_grad, network.parameters()), lr=lr,
                            weight_decay=weight_decay)
        super(MetaLearnerRegression, self).__init__(network, opt, self._on_batch_end)
        self.is_fitted = False
        self.writer = None
        self.plain_writer = None
        self.train_step = 0
        self.test_step = 0
        self.is_eval = False
        self.eval_params = dict()

        self.dataset_name = None

    def _compute_aux_return_loss(self, y_preds, y_tests):

        loss = torch.mean(torch.stack([mse_loss(y_pred, y_test)
                                       for y_pred, y_test in zip(y_preds, y_tests)]))

        return loss, dict(mse=loss)

    def _fit_batch(self, x, y, *, callback=Callback(), step=None, return_pred=False):
        self.optimizer.zero_grad()

        loss_tensor, metrics, pred_y = self._compute_loss_and_metrics(
            x, y, return_loss_tensor=True, return_pred=return_pred
        )
        if hasattr(self, 'grad_flow'):
            self.grad_flow.on_backward_start(step, loss_tensor)
        loss_tensor.backward()

        callback.on_backward_end(step)
        self.optimizer.step()

        loss = float(loss_tensor)
        return loss, metrics, pred_y

    def _on_batch_end(self, y_preds, y_tests):
        loss, scalars = self._compute_aux_return_loss(y_preds, y_tests)

        t = self.train_step if self.model.training else self.test_step
        # tag = 'train' if self.model.training else 'val'

        if self.writer is not None:
            tag = 'train' if self.model.training else 'valid'
            for k, v in scalars.items():
                self.writer.add_scalar(f'{tag}/{k}', to_unit(v), t)
                self.plain_writer.write(tag, k, to_unit(v), t)

        if torch.isnan(loss).any():
            raise Exception(f'{self.__class__.__name__}: Loss goes NaN')

        if self.model.training:
            self.train_step += 1
        else:
            self.test_step += 1
        return loss

    def fit(self, meta_train, meta_valid, meta_test=None, n_epochs=100, steps_per_epoch=100, log_filename=None,
            mse_filename=None, checkpoint_filename=None, tboard_folder=None, grads_inspect_dir=None,
            graph_flow_filename=None, do_early_stopping=True, mse_test=False, config=None):

        if hasattr(self.model, 'is_eval'):
            self.model.is_eval = False
        self.is_eval = False

        try:
            self.model.filename = log_filename[:-8]
        except:
            self.model.filename = 'test'

        self.steps_per_epoch = steps_per_epoch

        callbacks = [ReduceLROnPlateau(patience=2, factor=1 / 2, min_lr=1e-6, verbose=True),
                     BestModelRestore(verbose=True)]
        if do_early_stopping:
            callbacks.append(EarlyStopping(patience=10, verbose=False))

        if log_filename:
            callbacks.append(CSVLogger(log_filename, batch_granularity=False, separator='\t'))

        if mse_test:
            callbacks.append(MseMetaTest(meta_test=meta_test, filename=mse_filename, periodicity='epoch'))

        if checkpoint_filename:
            callbacks.append(ModelCheckpoint(checkpoint_filename, monitor='val_loss', save_best_only=True,
                                             temporary_filename=checkpoint_filename + 'temp'))

        if tboard_folder is not None:
            self.writer = SummaryWriter(tboard_folder)
            self.plain_writer = PlainWriter(tboard_folder)

        self.fit_generator(meta_train, meta_valid,
                           epochs=n_epochs,
                           steps_per_epoch=steps_per_epoch,
                           validation_steps=steps_per_epoch,
                           callbacks=callbacks,
                           verbose=True)
        self.is_fitted = True

        if self.plain_writer is not None:
            self.plain_writer.close()

        return self

    def _score_episode(self, episode, y_test, metrics, return_data=False, **kwargs):
        with torch.no_grad():
            x_train, y_train = episode['Dtrain']
            y_pred = self.model([episode])[0]

            if isinstance(y_pred, list):
                y_pred = y_pred[0]

            if self.model.return_var:
                y_pred, _ = y_pred
            else:
                y_pred = y_pred
        res = dict(test_size=y_pred.shape[0], train_size=y_train.shape[0])
        res.update({metric.__name__: to_unit(metric(y_pred, y_test)) for metric in metrics})

        if return_data:
            x_test, y_test = episode['Dtest']

            data = {
                'x_train': x_train,
                'y_train': y_train,
                'x_test': x_test,
                'y_test': y_test,
                'y_pred': y_pred
            }

            return res, data

        return res

    def _actvely_score_episode(self, episode, y_test, metrics, return_data=False, budget=20, **kwargs):
        if not self.model.return_var:
            raise Exception("This model can not be use for active learning")

        with torch.no_grad():
            x_train, y_train = episode['Dtrain']
            x_test, y_test = episode['Dtest']
            all_steps = self.model.forward_active([episode], return_all_steps=True, budget=budget)
            all_steps_random = self.model.forward_active([episode], return_all_steps=True, budget=budget, no_strategy=True)

        res = dict(test_size=y_test.shape[0], train_size=y_train.shape[0])
        for i, step in enumerate(all_steps):
            y_pred_active, _ = step[0]
            y_pred_random, _ = all_steps_random[i][0]
            res.update(
                {metric.__name__ + '_active_step' + str(i): to_unit(metric(y_pred_active, y_test)) for metric in metrics})
            res.update(
                {metric.__name__ + '_random_step' + str(i): to_unit(metric(y_pred_random, y_test)) for metric in metrics})

        return res

    def make_prediction(self, episodes):
        return self.model.forward(episodes)

    def evaluate(self, metatest, metrics=[mse_loss], tboard_folder=None, return_data=False, active_eval=False,
                 **kwargs):
        """
        Todo: save x_train, y_train, x_test, y_test & y_pred (test)

        Parameters
        ----------
        metatest
        metrics
        tboard_folder

        Returns
        -------

        """
        data = dict()

        if hasattr(self, 'model') and hasattr(self, 'writer'):
            self.model.eval()
            assert len(metrics) >= 1, "There should be at least one valid metric"
            if tboard_folder is not None:
                self.writer = SummaryWriter(tboard_folder)

        res_per_task = dict(name=dict(), error=dict())
        for batch in metatest:
            for episode, y_test in zip(*batch):
                ep_idx = episode['idx']
                try:
                    if active_eval:
                        scores = self._actvely_score_episode(episode, y_test, metrics, return_data=True)
                        if return_data:
                            data[ep_idx]['data'].append(dict())
                    else:
                        if return_data:
                            if ep_idx not in data:
                                data[ep_idx] = dict(name=metatest.dataset.tasks_filenames[ep_idx], data=[])
                            scores, episode_data = self._score_episode(episode, y_test, metrics, return_data=True)
                            data[ep_idx]['data'].append(episode_data)
                        else:
                            scores = self._score_episode(episode, y_test, metrics)
                except MemoryError as error:
                    res_per_task['error'][ep_idx] = True
                    print('A MemoryError happens')
                    continue

                for metric, m_value in scores.items():
                    if metric not in res_per_task:
                        res_per_task[metric] = dict()
                    if ep_idx not in res_per_task["name"]:
                        res_per_task[metric][ep_idx] = []
                    res_per_task[metric][ep_idx].append(m_value)
                res_per_task['name'][ep_idx] = metatest.dataset.tasks_filenames[ep_idx]
                res_per_task['error'][ep_idx] = (res_per_task['error'].get(ep_idx, False) or False)

        if return_data:
            return res_per_task, data

        return res_per_task

    def load(self, checkpoint_filename):
        self.load_weights(checkpoint_filename)
        self.is_fitted = True

    @staticmethod
    def static_load(checkpoint_filename, optimizer, loss_function, metrics=[]):
        model = torch.load(checkpoint_filename)
        return MetaLearnerRegression(model, optimizer, loss_function, metrics)
