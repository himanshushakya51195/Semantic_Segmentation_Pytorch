import os
import argparse
import pathlib
from dataset import SegmentationDataset
from SegTrainer import SegTrainingModule2
from trainmodule import TrainModuleRunner


def run_from_scratch(model_name: str,
                     epochs: int,
                     learning_rate: float,
                     data_dir: pathlib.Path,
                     batch_size: int,
                     workers: int,
                     val_fraction: float,
                     test_fraction: float,
                     architecture: str,
                     classes: int,
                     channels: int,
                     encoder_name: str,
                     encoder_weights: str,
                     encoder_depth: int,
                     model_save_dir,
                     **_):
    dataset = SegmentationDataset(data_dir)
    training_module = SegTrainingModule2.from_scratch(lr=learning_rate,
                                                      model_arch=architecture,
                                                      classes=classes,
                                                      in_channels=channels,
                                                      encoder_name=encoder_name,
                                                      encoder_weights=encoder_weights,
                                                      encoder_depth=encoder_depth)
    train_runner = TrainModuleRunner(
        training_module=training_module,
        n_epochs=epochs,
        dataset=dataset,
        batch_size=batch_size,
        n_workers=workers,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        model_name=model_name,
        model_root_dir=model_save_dir
    )
    train_runner.run()


def run_from_transfer_learning(model_name: str,
                               epochs: int,
                               learning_rate: float,
                               lr_decay_rate: float,
                               lr_step_size: int,
                               data_dir: pathlib.Path,
                               batch_size: int,
                               workers: int,
                               val_fraction: float,
                               test_fraction: float,
                               init_model: pathlib.Path,
                               init_hparams: pathlib.Path,
                               model_root,
                               **_):
    dataset = SegmentationDataset(data_dir)
    training_module = SegTrainingModule2.from_transfer_learning(lr=learning_rate,
                                                                lr_decay_rate=lr_decay_rate,
                                                                lr_step_size=lr_step_size,
                                                                init_state_path=init_model,
                                                                init_hparams_path=init_hparams)

    train_runner = TrainModuleRunner(
        training_module=training_module,
        n_epochs=epochs,
        dataset=dataset,
        batch_size=batch_size,
        n_workers=workers,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        model_name=model_name,
        root_dir=model_root)
    train_runner.run()


def get_run_function(train_mode: str):
    return globals()['run_' + 'from_' + train_mode]


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(prog='Segmentation_Model')

    # model training on scratch or transfer learning mode

    train_subsubparser = argparser.add_subparsers(dest='train_mode', help='which training mode to use')
    scratch_mode = train_subsubparser.add_parser('scratch')
    transfer_mode = train_subsubparser.add_parser('transfer_learning')

    # decalre name
    scratch_mode.add_argument('--model-name', '-n', required=True, type=str,
                              help='A name for a model')
    transfer_mode.add_argument('--model-name', '-n', required=True, type=str,
                               help='A name for a model')

    # declare destination directory
    scratch_mode.add_argument('--model-save-dir', '-d', required=True, type=dir_path,
                              help='directory to save scratch model')
    transfer_mode.add_argument('--model-save-dir', '-d', required=True, type=dir_path,
                               help='directory to save tl model')

    # both from scratch and tranfer learning requires 2 categories of arguments : Train, Data and Model

    # scratch group
    from_scratch_train_group = scratch_mode.add_argument_group(title='TRAIN options')
    from_scratch_data_group = scratch_mode.add_argument_group(title='DATA options')
    from_scratch_model_group = scratch_mode.add_argument_group(title='MODEL options')

    # transfer group
    transfer_learning_train_group = transfer_mode.add_argument_group(title='TRAIN options')
    transfer_learning_data_group = transfer_mode.add_argument_group(title='DATA options')
    transfer_learning_model_group = transfer_mode.add_argument_group(title='MODEL options')

    # epochs and learning rate for scratch training
    # for scratch training epochs = 100 and lr=0.0001

    from_scratch_train_group.add_argument('--epochs', '-ep', type=int, required=False, default=100,
                                          help='Number of epochs')
    from_scratch_train_group.add_argument('--learning-rate', '-lr', type=float, required=False, default=0.0001,
                                          help='Learning rate')

    # epochs, lr, lr-decay-rate, lr-step-size  for transfer learning
    # default values : epochs=30, lr=0.0001, lr_decay_rate=0.1, lr_step_size=3

    transfer_learning_train_group.add_argument('--epochs', '-ep', type=int, required=False, default=30,
                                               help='Number of epochs')
    transfer_learning_train_group.add_argument('--learning-rate', '-lr', type=float, required=False, default=0.0001,
                                               help='Learning Rate')
    transfer_learning_train_group.add_argument('--lr-decay-rate', '-lrd', type=float, required=False, default=0.5,
                                               help="learning scheduler decay rate")
    transfer_learning_train_group.add_argument('--lr-step-size', '-lss', type=int, required=False, default=5,
                                               help="learning scheduler step size")

    # all data groups arguments include data_dir, batch_size=4, workers=0, val_fraction=0.1, test_fraction=0.1
    # data_dir has no default value

    # data_dir
    from_scratch_data_group.add_argument('--data-dir', '-dd', type=dir_path, required=True,
                                         help='Training images and labels directory')
    transfer_learning_data_group.add_argument('--data_dir', '-dd', type=dir_path, required=True,
                                              help='Training images and labels directory')

    # batch size
    from_scratch_data_group.add_argument('--batch-size', '-bs', type=int, required=False, default=4,
                                         help='Batch size of data loader')
    transfer_learning_data_group.add_argument('--batch-size', '-bs', type=int, required=False, default=4,
                                              help='Batch size of data loader')

    # worker
    from_scratch_data_group.add_argument('--workers', '-wk', type=int, required=False, default=0,
                                         help='Number of workers in data loader')
    transfer_learning_data_group.add_argument('--worker', '-wk', type=int, required=False, default=0,
                                              help='Number of workers in data loader')

    # validation fraction
    from_scratch_data_group.add_argument('--val-fraction', '-vf', type=float, required=False, default=0.1,
                                         help='Fraction of data used for validation')
    transfer_learning_data_group.add_argument('--val-fraction', '-vf', type=float, required=False, default=0.1,
                                              help='Fraction of data used for validation')

    # test fraction
    from_scratch_data_group.add_argument('--test-fraction', '-tf', type=float, required=False, default=0.1,
                                         help='Fraction of data used for test')
    transfer_learning_data_group.add_argument('--test-fraction', '-tf', type=float, required=False, default=0.1,
                                              help='Fraction of data used for test')

    # finetune model requires init_weights and init_hparams_path
    # both args required with no default value

    transfer_learning_model_group.add_argument('--init-model', '-im', required=True, type=dir_path,
                                               help='File path to initial weights (previously trained model)')

    transfer_learning_model_group.add_argument('--init-hparams', '-ip', required=True, type=dir_path,
                                               help='File path to hyperparameters')

    # add scratch model args

    from_scratch_model_group.add_argument('--architecture', '-nn', type=str, required=False, default='Unet',
                                          help='Model architecture')
    from_scratch_model_group.add_argument('--classes', '-cl', type=int, required=False, default=2,
                                          help='Number of classes')
    from_scratch_model_group.add_argument('--channels', '-ch', type=int, required=False, default=3,
                                          help='Number of channels in input image')
    from_scratch_model_group.add_argument('--encoder-name', '-bb', type=str, required=False, default='xception',
                                          help='Encoder/backbone of the model')
    from_scratch_model_group.add_argument('--encoder-weights', '-bw', type=str, required=False, default=None,
                                          help='Initial weights of encoder/backbone')
    from_scratch_model_group.add_argument('--encoder-depth', '-bd', type=int, required=False, default=5,
                                          help='Number of stages in encoder/backbone')

    args = argparser.parse_args()

    run_func = get_run_function(args.train_mode)
    run_func(**vars(args))
