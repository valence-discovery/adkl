from poutyne.framework.callbacks import Callback
import torch

from torch.nn.functional import mse_loss


class MseMetaTest(Callback):
    """
    Callback object that estimates the MSE loss on the meta test set after each batch or epoch.
    """

    def __init__(self, meta_test, filename, periodicity='epoch'):
        """

        Parameters
        ----------
        meta_test: meta dataset.
            The meta-dataset used for testing.
        filename: str
            The name of the csv file to populate.
        periodicity: str, {'batch', 'epoch'}
            The periodicity of the test.
        """

        assert periodicity in {'batch', 'epoch'}, "Periodicity should be either 'batch' or 'epoch'."

        super(MseMetaTest, self).__init__()

        self.meta_test = meta_test

        self.periodicity = periodicity

        self.filename = filename
        self.writer = None

    def on_train_begin(self, logs):
        self.writer = open(self.filename, 'a')

        if self.periodicity == 'batch':
            self.writer.write('epoch, batch, mse\n')
        else:
            self.writer.write('epoch, mse\n')

    def get_mse(self):
        """
        Gets the mean squared error on the meta test set.

        Returns
        -------
        mse: torch.Tensor
            1D tensor representing the mean squared error on the meta test set.
        """

        predictions = []
        targets = []

        for i, batch in enumerate(self.meta_test):
            for j, (episode, y_test) in enumerate(zip(*batch)):

                if self.model.model.return_var:
                    y_pred, _ = self.model.make_prediction([episode])[0]
                else:
                    y_pred = self.model.make_prediction([episode])[0]

                y_pred = y_pred.detach()

                predictions.append(y_pred)
                targets.append(y_test)

                if j > 20:
                    break

            if i > 40:
                break

        predictions = torch.cat(tuple(predictions))
        targets = torch.cat(tuple(targets))

        mse = mse_loss(predictions, targets)

        return mse

    def on_batch_end(self, batch, logs):

        if self.periodicity == 'batch':
            mse = self.get_mse()
            epoch = logs['epoch']
            self.writer.write(f'{epoch}, {batch}, {mse}\n')

    def on_epoch_end(self, epoch, logs):

        if self.periodicity == 'epoch':
            mse = self.get_mse()
            self.writer.write(f'{epoch}, {mse}\n')

    def on_train_end(self, logs):
        self.writer.close()
