from importlib import import_module
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

class Data:
    def __init__(self, args):
        kwargs = {
            'collate_fn': default_collate,
            'pin_memory': True
        }

        self.loader_train = None
        if not args.test_only:
            module_train = import_module('data.' + args.data_train.lower())
            trainset = getattr(module_train, args.data_train)(args)
            self.loader_train = DataLoader(
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                **kwargs
            )

        module_test = import_module('data.' + args.data_test.lower())
        testset = getattr(module_test, args.data_test)(args, train=False)

        self.loader_test = DataLoader(
            testset,
            batch_size=1,
            shuffle=False,
            **kwargs
        )
