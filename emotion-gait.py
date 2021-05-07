import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib
import random
import time
import argparse
import torch.utils.data
import matplotlib.pyplot as plt
import h5py
import sys
import os
import bz2
import json
import datetime
from modules.dataset_egait import *
from modules.models import *
from modules.tensorboard_utils import CustomSummaryWriter

modelSaveFile = 'model.pt'

modelsMap = {
    "ModelRNN": ModelRNN,
}

parser = argparse.ArgumentParser(add_help=False)

parser.add_argument('--id', default=0, type=int)
parser.add_argument('--run-name', default=f'run_{time.time()}', type=str)
parser.add_argument('--sequence-name', default=f'seq_default', type=str)
parser.add_argument('--learning-rate', default=1e-3, type=float)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--lstm-layers', default=2, type=int)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--use-cuda', action='store_true')
parser.add_argument('--save-model', action='store_true')
parser.add_argument('--load-model', action='store_true')
parser.add_argument('--save-artefacts', action='store_true')
parser.add_argument("--dataset-features", "-f", nargs="+", default=["features.h5", "features_ELMD.h5"],
                    help="Dataset file names list. Default: features.h5, features_ELMD.h5")
parser.add_argument("--dataset-labels", "-l", nargs="+", default=["labels.h5", "labels_ELMD.h5"],
                    help="Dataset labels file list. Default: labels.h5, labels_ELMD.h5")
parser.add_argument('--remove-zeromove', action='store_true',
                    help="Remove frames which does not contain movement.")
parser.add_argument('--center-root', action='store_true',
                    help="Center features so that root point is always at coordinates 0,0,0." +
                    " This already done for STEP dataset, but not for ELMD dataset.")
parser.add_argument('--rotate-y', action='store_true',
                    help="Randomly rotate features around Y axis")
parser.add_argument('--scale', action='store_true',
                    help="Randomly scale features making picture 'bigger' or 'smaller'")
parser.add_argument('--equalize', action='store_true',
                    help="Copy required gaits so all emotions are equally covered")
parser.add_argument('--drop-elmd-frames', action='store_true',
                    help="Drop every 2nd frame from ELMD dataset so movement speed looks more in line with STEP dataset")
parser.add_argument('--normalize-gait-sizes', action='store_true',
                    help="Normalize sizes so that all feature sizes are the same")
parser.add_argument('--model', default="ModelRNN", type=str, choices=modelsMap.keys())
parser.add_argument('--hidden-size', default=256, type=int)
parser.add_argument('--save-epoch-gait-video', action='store_true',
                    help="Save one random video to from train epoch to tensorboard results.")


if 'COLAB_GPU' in os.environ:
    args = parser.parse_args(args=[])
else:
    args = parser.parse_args()

random.seed()

# Dynamically loading module won't check if such model exists
# Model = getattr(__import__('modules_core.' + args.model, fromlist=['Model']), 'Model')
# model = Model()
modelCallback = modelsMap[args.model]

if args.save_artefacts:
    from modules.file_utils import *
    import modules.dict_to_obj
    from modules.csv_utils_2 import *

    path_sequence = f'./results/{args.sequence_name}'
    args.run_name += ('-' + datetime.datetime.utcnow().strftime(f'%y-%m-%d--%H-%M-%S'))
    path_run = f'./results/{args.sequence_name}/{args.run_name}'
    path_artefacts = f'./artefacts/{args.sequence_name}/{args.run_name}'
    FileUtils.createDir(path_run)
    FileUtils.createDir(path_artefacts)
    FileUtils.writeJSON(f'{path_run}/args.json', args.__dict__)

    CsvUtils2.create_global(path_sequence)
    CsvUtils2.create_local(path_sequence, args.run_name)
    print("Done creating artefacts directories")
    summary_writer = CustomSummaryWriter(
        logdir=path_artefacts
    )
    print("Initialized tensorboard summary writer")

data_loader_train = torch.utils.data.DataLoader(
    dataset=DatasetEGait(is_train=True,
                         dataset_files=[args.dataset_features, args.dataset_labels],
                         remove_zeromove=args.remove_zeromove,
                         center_root=args.center_root,
                         rotate_y=args.rotate_y,
                         scale=args.scale,
                         equalize=args.equalize,
                         drop_elmd_frames=args.drop_elmd_frames,
                         normalize_gait_sizes=args.normalize_gait_sizes
                         ),
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True
)

data_loader_test = torch.utils.data.DataLoader(
    dataset=DatasetEGait(is_train=False,
                         dataset_files=[args.dataset_features, args.dataset_labels],
                         remove_zeromove=args.remove_zeromove,
                         center_root=args.center_root,
                         rotate_y=args.rotate_y,
                         scale=args.scale,
                         equalize=args.equalize,
                         drop_elmd_frames=args.drop_elmd_frames,
                         normalize_gait_sizes=args.normalize_gait_sizes
                         ),
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=True
)
print("Dataset loaded")


if torch.cuda.is_available() and args.use_cuda:
    print("Using CUDA. Device count: %d" % torch.cuda.device_count())
    args.device = 'cuda'
else:
    args.device = 'cpu'

model = modelCallback(hidden_size=args.hidden_size, lstm_layers=args.lstm_layers)

if args.load_model:
    if os.path.isfile(modelSaveFile):
        print("Loaded model from %s" % modelSaveFile)
        model.load_state_dict(torch.load(modelSaveFile))
        model.eval()

model.to(args.device)

# if args.device == 'cuda' and torch.cuda.device_count() > 1:
#    print("Parallelling data")
#    model = torch.nn.DataParallel(model, dim=0)  # Splittable dimension

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.learning_rate
    # betas=(0.9, 0.999),
    # eps=1e-8,
    # weight_decay=0
)

metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'acc',
        'time'
    ]:
        metrics[f'{stage}_{metric}'] = []

bestLoss = 1e20
for epoch in range(1, args.epochs + 1):
    currentLoss = 1e20
    metrics_epoch = {key: [] for key in metrics.keys()}
    conf_matrix = np.zeros((4, 4))
    for data_loader in [data_loader_train, data_loader_test]:

        start_time = datetime.datetime.utcnow()

        stage = 'train'
        if data_loader == data_loader_test:
            stage = 'test'
            model = model.eval()
        else:
            model = model.train()

        for x, lengths, y_idx in data_loader:
            if args.save_epoch_gait_video and stage == 'train':
                ran = random.randint(0, x.size(0))
                gaitFrames = x[ran]
                gaitName = y_idx[ran]
            x = x.to(args.device)
            y_idx = y_idx.to(args.device).squeeze()

            y_prim = model.forward(x, lengths)

            idxes = torch.arange(x.size(0)).to(args.device)
            loss = -torch.mean(torch.log(y_prim[idxes, y_idx] + 1e-8))
            y_idx_prim = torch.argmax(y_prim, dim=1)

            acc = torch.mean((y_idx == y_idx_prim) * 1.0)
            metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())
            metrics_epoch[f'{stage}_acc'].append(acc.cpu().item())
            metrics_epoch[f'{stage}_time'].append((datetime.datetime.utcnow() - start_time).total_seconds())

            # y = torch.softmax(torch.randn((32, class_count)), dim=1)
            y_prim = torch.softmax(y_prim, dim=1).cpu()
            y_prim_idx = torch.argmax(y_prim, dim=1).data.numpy()
            for idx in range(y_idx.shape[0]):
                conf_matrix[y_idx[idx], y_idx_prim[idx]] += 1

            if data_loader == data_loader_train:
                currentLoss = loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        metrics_strs = []
        for key in metrics_epoch.keys():
            if stage in key:
                value = np.mean(metrics_epoch[key])
                metrics[key].append(value)
                metrics_strs.append("%-10s %-5.2f" % (key, value))

        print("epoch: %-3s stage: %-5s" % (epoch, stage), " ".join(metrics_strs))

    if args.save_artefacts:
        summary_writer.add_scalar(
            tag='train_loss',
            scalar_value=np.mean(metrics_epoch['train_loss']),
            global_step=epoch
        )

        summary_writer.add_scalar(
            tag='train_acc',
            scalar_value=np.mean(metrics_epoch['train_acc']),
            global_step=epoch
        )

        summary_writer.add_hparams(
            hparam_dict=args.__dict__,
            metric_dict={
                'train_loss': np.mean(metrics_epoch['train_loss']),
                'train_acc': np.mean(metrics_epoch['train_acc']),
                'test_loss': np.mean(metrics_epoch['test_loss']),
                'test_acc': np.mean(metrics_epoch['test_acc']),
                'train_time': np.mean(metrics_epoch['train_time']),
                'test_time': np.mean(metrics_epoch['test_time']),
            },
            name=args.run_name,
            global_step=epoch
        )

    fig, axs = plt.subplots(3, figsize=(6, 10), gridspec_kw={'height_ratios': [1, 1, 2]})
    fig.suptitle('Loss & Accuracy over epochs')
    axs[0].clear()
    axs[1].clear()
    axs[2].clear()
    plts = []
    c = 0

    class_count = 4
    for key, value in metrics.items():
        if 'time' in key:
            continue
        if 'loss' in key:
            plts += axs[0].plot(value, f'C{c}', label=key)
        else:
            plts += axs[1].plot(value, f'C{c}', label=key)
        c += 1
    axs[0].set(ylabel='loss')
    axs[1].set(xlabel='epoch', ylabel='acc')
    axs[0].legend(plts, [it.get_label() for it in plts])

    im = axs[2].imshow(conf_matrix, interpolation='nearest', cmap=plt.get_cmap('Greys'))
    plt.sca(axs[2])
    plt.xticks([0, 1, 2, 3], ['Angry', 'Neutral', 'Happy', 'Sad'])
    plt.yticks([0, 1, 2, 3], ['Angry', 'Neutral', 'Happy', 'Sad'])
    for x in range(class_count):
        for y in range(class_count):
            axs[2].annotate(
                str(round(100 * conf_matrix[x, y]/np.sum(conf_matrix[x]), 1)),
                xy=(y, x),
                horizontalalignment='center',
                verticalalignment='center',
                backgroundcolor='white'
            )
    plt.xlabel('Ground Truth')
    plt.ylabel('Predicted')

    if args.save_artefacts:
        CsvUtils2.add_hparams(
            path_sequence,
            args.run_name,
            args.__dict__,
            metrics,  # contains train + test metrics
            epoch
        )

        summary_writer.add_figure(
            tag='charts',
            figure=fig,
            global_step=epoch
        )

        if args.save_epoch_gait_video:
            # TODO Save animation in tensorboard.
            # vis = VisualizeGait(return_video=True)
            #
            # summary_writer.add_video(
            #     tag='gait',
            #     vid_tensor=vis.vizualize(gaitFrames, gaitName),
            #     global_step=epoch,
            #     fps=25
            # )
            pass

        # summary_writer.add_embedding(
        #     mat=embeddings,
        #     metadata=classes.tolist(),
        #     tag='embeddings',
        #     global_step=epoch
        # )

        summary_writer.flush()
    else:
        plt.draw()
        plt.pause(.001)

    if args.save_model:
        if currentLoss < bestLoss:
            bestLoss = currentLoss
            torch.save(model.state_dict(), modelSaveFile)
            print("Saved model to %s " % modelSaveFile)


if args.save_artefacts:
    summary_writer.close()
