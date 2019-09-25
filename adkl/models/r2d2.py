from .metakrr_sk import MetaKrrSKLearner


class R2D2Learner(MetaKrrSKLearner):
    def __init__(self, *args,  **kwargs):
        super(R2D2Learner, self).__init__(*args, **kwargs, kernel='linear', meta_dropout=0, normalize_kernel=False)


if __name__ == '__main__':
    pass
