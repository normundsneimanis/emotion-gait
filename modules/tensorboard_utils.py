from tensorboardX import SummaryWriter
from tensorboardX.summary import hparams
import numpy as np


class CustomSummaryWriter(SummaryWriter):
    def __init__(self, logdir=None, comment='', purge_step=None,
                 max_queue=10, flush_secs=120, filename_suffix='',
                 write_to_disk=True, log_dir=None, **kwargs):
        super().__init__(logdir, comment, purge_step, max_queue,
                         flush_secs, filename_suffix, write_to_disk,
                         log_dir, **kwargs)

    def add_hparams(self, hparam_dict=None, metric_dict=None, name=None, global_step=None):
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            self.add_scalar(k, v, global_step)


def save_tensorboard(writer, epoch, metrics_epoch, best_metrics, args_save, run_name):
    writer.add_scalar(
        tag='train_loss',
        scalar_value=np.mean(metrics_epoch['train_loss']),
        global_step=epoch
    )

    writer.add_scalar(
        tag='train_acc',
        scalar_value=np.mean(metrics_epoch['train_acc']),
        global_step=epoch
    )

    writer.add_scalar(
        tag='best_test_loss',
        scalar_value=best_metrics['best_test_loss'],
        global_step=epoch
    )

    writer.add_scalar(
        tag='best_train_loss',
        scalar_value=best_metrics['best_train_loss'],
        global_step=epoch
    )

    writer.add_hparams(
        hparam_dict=args_save,
        metric_dict={
            'train_loss': np.mean(metrics_epoch['train_loss']),
            'train_acc': np.mean(metrics_epoch['train_acc']),
            'test_loss': np.mean(metrics_epoch['test_loss']),
            'test_acc': np.mean(metrics_epoch['test_acc']),
            'train_time': np.mean(metrics_epoch['train_time']),
            'test_time': np.mean(metrics_epoch['test_time']),
        },
        name=run_name,
        global_step=epoch
    )