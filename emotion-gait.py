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
import copy
from modules.dataset_egait import *
from modules.models import *
from modules.model_transformer import *
from modules.model_resnet import *
from modules.loss_history import LossHistory
from sklearn.metrics import f1_score


modelsMap = {
    "ModelRNN": ModelRNN,
    "Model-P-LSTM": ModelLSTM,
    "Transformer": Transformer,
    "SotaLSTM": ModelSotaLSTM,
    "ModelConvolutional": ModelConvolutional,
    "Transformer2": Transformer2,
    "ResNet16": ResNet16,
    "ResNet36": ResNet36,
    "ResNet16BN": ResNet16BN,
    "ResNet36BN": ResNet36BN,
    "ResNet54": ResNet54,
    "ResNet100": ResNet100,
    "ResNet150": ResNet150,
}

parser = argparse.ArgumentParser(add_help=False)

parser.add_argument('--id', default=0, type=int)
parser.add_argument('--run-name', type=str,
                    default=f'run_{time.time()}-' + datetime.datetime.utcnow().strftime(f'%y-%m-%d--%H-%M-%S'))
parser.add_argument('--sequence-name', default=f'seq_default', type=str)
parser.add_argument('--learning-rate', default=1e-3, type=float)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--lstm-layers', default=2, type=int)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--use-cuda', action='store_true')
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
parser.add_argument('--drop-elmd-frames', action='store_true',
                    help="Drop every 2nd frame from ELMD dataset so movement speed looks more in line with STEP dataset")
parser.add_argument('--normalize-gait-sizes', action='store_true',
                    help="Normalize sizes so that all feature sizes are the same")
parser.add_argument('--model', default="ModelRNN", type=str, choices=modelsMap.keys())
parser.add_argument('--hidden-size', default=256, type=int)
parser.add_argument('--save-epoch-gait-video', action='store_true',
                    help="Save one random video to from train epoch to results.")
parser.add_argument('--cuda-device-num', default=-1, type=int)
parser.add_argument('--save-best', action='store_true',
                    help="Save best results in tensorboard")
parser.add_argument('--overfit-exit-percent', default=0, type=int,
                    help="Set percent of epochs. 0 to disable")
parser.add_argument('--regularization-lambda', default=1e-4, type=float,
                    help="Lambda, regularization coefficient. Start with L1:1e-5 L2:1e-3 L3:1e-5")
# Model-P-LSTM-LayerNorm: L3: 1e-7
# SotaLSTM w/ layernorm: L1: 1e-7, L2: 1e-4 L3: 1e-7
parser.add_argument('--regularization-alpha', default=0.95, type=float,
                    help="Elastic Net alpha term. Range [0-1] between L1 and L2.")
parser.add_argument('--regularization-l', default=0, type=int, choices=[0, 1, 2, 3],
                    help="Regularization. 0 - off, 1 - Lasso, 2 - Ridge, 3 - Elastic Net.")
parser.add_argument('--p-lstm-alpha', default=1e-3, type=float,
                    help="Leak rate is active in the closed phase, and plays a similar role as the leak in a "
                         "parametric 'leaky' rectified linear unit")
parser.add_argument('--p-lstm-tau-max', default=3.0, type=float,
                    help="Phase period Tau controls the real-time period of the oscillation")
parser.add_argument('--p-lstm-r-on', default=5e-2, type=float, help="Ratio of the open period to the total period Tau")
parser.add_argument('--loss', default="wce", type=str, choices=["wce", "acc", "dice", "focal"],
                    help="Loss function: Weighted Cross Entropy, Accuracy or Dice loss")
parser.add_argument('--layernorm', action='store_true',
                    help="Use layer normalization")
parser.add_argument('--memmap', action='store_true',
                    help="Use memmap for storing dataset")
parser.add_argument('--memmap-only', action='store_true')
parser.add_argument('--transformer-heads', default=8, type=int,
                    help="Number of Transformer heads")
parser.add_argument('--transformer-depth', default=6, type=int,
                    help="Number of Transformer heads")
parser.add_argument('--transformer-embed-dim', default=64, type=int,
                    help="Number of Transformer heads")
parser.add_argument('--disable-early-exit', action='store_true')
parser.add_argument('--affective-features', action='store_true')


if 'COLAB_GPU' in os.environ:
    args = parser.parse_args(args=[])
else:
    args = parser.parse_args()

if args.regularization_l == 3 and (args.regularization_lambda > 1.0 or args.regularization_lambda < 0.0):
    print("Error: --regularization-lambda cannot be >1.0 for --regularization-l 3 (Elastic Net). Exiting")
    exit(1)

if (args.dataset_features != ["features.h5", "features_ELMD.h5"] and
        args.dataset_labels == ["labels.h5", "labels_ELMD.h5"]):
    args.dataset_labels = []

for n, i in enumerate(args.dataset_labels):
    if i == 'None':
        args.dataset_labels[n] = None

random.seed()

# Dynamically loading module won't check if such model exists
# Model = getattr(__import__('modules_core.' + args.model, fromlist=['Model']), 'Model')
# model = Model()
modelCallback = modelsMap[args.model]

if args.save_artefacts:
    total_start_time = datetime.datetime.utcnow()

    from modules.file_utils import *
    import modules.dict_to_obj
    from modules.csv_utils_2 import *
    from modules.tensorboard_utils import *

    path_sequence = f'./results/{args.sequence_name}'
    path_run = f'./results/{args.sequence_name}/{args.run_name}'
    path_artefacts = f'./artefacts/{args.sequence_name}/{args.run_name}'
    path_artefacts_best = f'./artefacts/{args.sequence_name}/{args.run_name}-best'
    FileUtils.createDir(path_run)
    FileUtils.createDir(path_artefacts)
    FileUtils.writeJSON(f'{path_run}/args.json', args.__dict__)

    CsvUtils2.create_global(path_sequence)
    CsvUtils2.create_local(path_sequence, args.run_name)
    print("Done creating artefacts directories")
    summary_writer = CustomSummaryWriter(
        logdir=path_artefacts
    )
    if args.save_best:
        best_writer = CustomSummaryWriter(
            logdir=path_artefacts_best
        )
    print("Initialized tensorboard summary writer")
    args_save = copy.deepcopy(args.__dict__)
    args_save['dataset-features'] = " ".join(args.dataset_features)
    args_save['dataset-labels'] = " ".join(args.dataset_labels)

data_loader_train = torch.utils.data.DataLoader(
    dataset=DatasetEGait(is_train=True,
                         dataset_files=[args.dataset_features, args.dataset_labels],
                         remove_zeromove=args.remove_zeromove,
                         center_root=args.center_root,
                         rotate_y=args.rotate_y,
                         scale=args.scale,
                         drop_elmd_frames=args.drop_elmd_frames,
                         normalize_gait_sizes=args.normalize_gait_sizes,
                         memmap=args.memmap,
                         memmap_only=args.memmap_only,
                         affective_features=args.affective_features,
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
                         drop_elmd_frames=args.drop_elmd_frames,
                         normalize_gait_sizes=args.normalize_gait_sizes,
                         memmap=args.memmap,
                         memmap_only=args.memmap_only,
                         affective_features=args.affective_features
                         ),
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=True
)
print("Dataset loaded")


if torch.cuda.is_available() and args.use_cuda:
    print("Using CUDA. Device count: %d" % torch.cuda.device_count())
    if args.cuda_device_num != -1:
        args.device = torch.device('cuda:%d' % args.cuda_device_num)
        print("Forcing device %d" % args.cuda_device_num)
    else:
        args.device = 'cuda'
else:
    args.device = 'cpu'

if args.model == 'Model-P-LSTM':
    model = modelCallback(hidden_size=args.hidden_size, lstm_layers=args.lstm_layers, device=args.device,
                          alpha=args.p_lstm_alpha, tau_max=args.p_lstm_tau_max, r_on=args.p_lstm_r_on,
                          layernorm=args.layernorm)
elif args.model == 'Transformer2':
    model = modelCallback(hidden_size=args.hidden_size, lstm_layers=args.lstm_layers, device=args.device,
                          layernorm=args.layernorm, dim=args.transformer_embed_dim, depth=args.transformer_depth,
                          heads=args.transformer_heads)
else:
    model = modelCallback(hidden_size=args.hidden_size, lstm_layers=args.lstm_layers, device=args.device,
                          layernorm=args.layernorm)

total_params = 0
for parameter in model.parameters():
    param_count = 1
    for p in parameter.shape:
        param_count *= p
    total_params += param_count

print("Number of params: %d" % total_params)

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
        'time',
        'lp1',
        'reg',
        'f1score'
    ]:
        metrics[f'{stage}_{metric}'] = []

best_metrics = {
    'best_train_loss': 1e20,
    'best_test_loss': 1e20,
    'best_train_acc': 0,
    'best_test_acc': 0,
    'best_f1': 0,
}
testAccWorseCount = 0
testLossWorseCount = 0

# Class weights using Divison method
class_weights = (0.5 / np.array(DatasetEGait.labelsCount)) * sum(DatasetEGait.labelsCount)
class_weights = torch.tensor(class_weights).clone().detach().to(args.device)

lossHistory = LossHistory()

for epoch in range(1, args.epochs + 1):
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
                ran = random.randint(0, x.size(0)-1)
                gaitFrames = x[ran][0:lengths[ran]]
                gaitName = y_idx[ran]
            x = x.to(args.device)
            y_idx = y_idx.to(args.device).squeeze()

            if args.model == 'Transformer':
                idxes = torch.argsort(lengths, descending=True)
                lengths = lengths[idxes]
                max_len = int(lengths.max())
                x = x[idxes, :max_len]  # (Batch, Seq, Features)
                y_idx = y_idx[idxes]
                lengths2 = torch.ones(len(y_idx)).type(torch.int64)
                y_packed = pack_padded_sequence(y_idx.view(args.batch_size, 1, 1), lengths2, batch_first=True)

            y_prim = model.forward(x, lengths)

            idxes = torch.arange(x.size(0)).to(args.device)

            regularization_term = torch.tensor(0., requires_grad=True).to(args.device)
            if args.regularization_l:
                for param in model.parameters():
                    if args.regularization_l == 3:  # Elastic Net
                        regularization_term += ((1.0 - args.regularization_alpha) *
                                                torch.norm(param, p=1) +
                                                args.regularization_alpha * torch.norm(param, p=2))
                    else:  # L1 or L2
                        regularization_term += torch.norm(param, p=args.regularization_l)

            # Regularization term
            loss_reg = (args.regularization_lambda * regularization_term)

            # Loss function
            if args.loss == "acc":  # Binary cross entropy
                loss1 = -torch.mean(torch.log(y_prim[idxes, y_idx] + 1e-8))  # Loss without weights
            elif args.loss == "wce":  # Weighted cross entropy
                loss1 = (-torch.mean(torch.log(y_prim[idxes, y_idx] + 1e-8) * class_weights[y_idx]))
            elif args.loss == "dice":
                loss1 = torch.mean(2 * (y_idx * y_prim[idxes, y_idx] + 1e-8) / (y_idx + y_prim[idxes, y_idx] + 1e-8))
            elif args.loss == "focal":
                gamma = 0.
                loss1 = torch.mean(-1 * (1-(y_prim[idxes, y_idx] + 1e-8))**gamma * torch.log(y_prim[idxes, y_idx] + 1e-8))

            # Combine loss with regularization term, if set
            loss = loss1 + loss_reg

            # Calculate accuracy and f-score
            y_idx_prim = torch.argmax(y_prim, dim=1)
            acc = torch.mean((y_idx == y_idx_prim) * 1.0)
            f1score = f1_score(y_idx.to('cpu'), y_idx_prim.to('cpu'), average='micro', zero_division=0)

            metrics_epoch[f'{stage}_lp1'].append(loss1.cpu().item())
            metrics_epoch[f'{stage}_reg'].append(loss_reg.cpu().item())
            metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())
            metrics_epoch[f'{stage}_acc'].append(acc.cpu().item())
            metrics_epoch[f'{stage}_f1score'].append(f1score.item())
            metrics_epoch[f'{stage}_time'].append((datetime.datetime.utcnow() - start_time).total_seconds())

            y_prim = torch.softmax(y_prim, dim=1).cpu()
            y_prim_idx = torch.argmax(y_prim, dim=1).data.numpy()
            for idx in range(y_idx.shape[0]):
                conf_matrix[y_idx[idx], y_idx_prim[idx]] += 1

            if data_loader == data_loader_train:
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

    lossHistory.addTrainLoss(np.mean(metrics_epoch['train_loss']))
    lossHistory.addTestLoss(np.mean(metrics_epoch['test_loss']))
    lossHistory.addTestAcc(np.mean(metrics_epoch['test_acc']))
    if np.mean(metrics_epoch['train_acc']) > best_metrics['best_train_acc']:
        best_metrics['best_train_acc'] = np.mean(metrics_epoch['train_acc'])
    if np.mean(metrics_epoch['test_f1score']) > best_metrics['best_f1']:
        best_metrics['best_f1'] = np.mean(metrics_epoch['test_f1score'])
    if np.mean(metrics_epoch['test_acc']) > best_metrics['best_test_acc']:
        best_metrics['best_test_acc'] = np.mean(metrics_epoch['test_acc'])
        testAccWorseCount = 0
    else:
        testAccWorseCount += 1
    if np.mean(metrics_epoch['train_loss']) < best_metrics['best_train_loss']:
        best_metrics['best_train_loss'] = np.mean(metrics_epoch['train_loss'])
    if np.mean(metrics_epoch['test_loss']) < best_metrics['best_test_loss']:
        best_metrics['best_test_loss'] = np.mean(metrics_epoch['test_loss'])
        testLossWorseCount = 0
    else:
        testLossWorseCount += 1

    if args.save_artefacts:
        save_tensorboard(summary_writer, epoch, metrics_epoch, best_metrics, args_save, args.run_name)
        if args.save_best and (testLossWorseCount == 0 or testAccWorseCount == 0):
            save_tensorboard(best_writer, epoch, metrics_epoch, best_metrics, args_save, args.run_name)

    fig, axs = plt.subplots(3, figsize=(6, 10), gridspec_kw={'height_ratios': [1, 1, 2]})
    fig.suptitle('Loss & Accuracy over epochs')
    axs[0].clear()
    axs[1].clear()
    axs[2].clear()
    plts = []
    c = 0

    class_count = 4
    for key, value in metrics.items():
        if 'loss' in key:
            plts += axs[0].plot(value, f'C{c}', label=key)
        elif 'acc' in key:
            plts += axs[1].plot(value, f'C{c}', label=key)
        else:
            continue
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
            args_save,
            {
                'total_minutes': "%.1f" %
                                 (float(((datetime.datetime.utcnow() - total_start_time).total_seconds())) / 60.0),
                'train_loss': np.mean(metrics_epoch['train_loss']),
                'train_acc': np.mean(metrics_epoch['train_acc']),
                'test_loss': np.mean(metrics_epoch['test_loss']),
                'test_acc': np.mean(metrics_epoch['test_acc']),
                'test_f1': np.mean(metrics_epoch['test_f1score']),
                **best_metrics
            },  # contains train + test metrics + best metrics
            epoch
        )

        summary_writer.add_figure(
            tag='charts',
            figure=fig,
            global_step=epoch
        )
        if args.save_best and (testLossWorseCount == 0 or testAccWorseCount == 0):
            best_writer.add_figure(
                tag='charts',
                figure=fig,
                global_step=epoch
            )

        if args.save_epoch_gait_video and epoch % 10 == 0:
            vis = VisualizeGait(return_video=True)
            with open(os.path.join(path_run, "%04d.html" % epoch), "w") as f:
                print(vis.vizualize(gaitFrames, gaitName), file=f)

        summary_writer.flush()
        if args.save_best:
            best_writer.flush()
    else:
        plt.draw()
        plt.pause(.001)

    if args.overfit_exit_percent > 0 and \
            (args.epochs >= 200 and
             (testAccWorseCount >= args.epochs/args.overfit_exit_percent or
              testLossWorseCount >= args.epochs/args.overfit_exit_percent)):
        print("Exiting because overfitting has been detected")
        break

    if not args.disable_early_exit and epoch % 10 == 0 and lossHistory.diverging():
        print("Exiting because loss is diverging")
        break


if args.save_artefacts:
    summary_writer.close()
    if args.save_best:
        best_writer.close()

print("Total run time: %.1f minutes" % (float(((datetime.datetime.utcnow() - total_start_time).total_seconds())) / 60.0))
