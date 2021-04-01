import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib
import torchvision
from tqdm import tqdm
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

modelSaveFile = 'model.pt'

parser = argparse.ArgumentParser(add_help=False)

parser.add_argument('--id', default=0, type=int)
parser.add_argument('--run-name', default=f'run_{time.time()}', type=str)
parser.add_argument('--sequence-name', default=f'seq_default', type=str)
parser.add_argument('--learning-rate', default=1e-3, type=float)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--lstm-layers', default=2, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--save-model', action='store_true')
parser.add_argument('--load-model', action='store_true')
parser.add_argument('--dataset-no-memory', action='store_false')
parser.add_argument('--load-memory-percent', default=70, type=int)

parser.add_argument('--hidden-size', default=256, type=int)

if 'COLAB_GPU' in os.environ:
    args = parser.parse_args(args=[])
else:
    args = parser.parse_args()


class DatasetEGait(torch.utils.data.Dataset):
    loaded = 0
    hf = 0
    gaits = 0
    lengths = 0
    labels = 0
    train_gaits = []
    test_gaits = []
    path = './emotion-gait'

    def __init__(self, is_train, in_memory=False, load_percent=50):
        self.is_train = is_train
        self.in_memory = in_memory

        if not DatasetEGait.loaded:
            self.loadDataSet()

        if is_train:
            self.gaitNames = DatasetEGait.train_gaits
        else:
            self.gaitNames = DatasetEGait.test_gaits

        if in_memory:
            loadMax = int(len(self.gaitNames) * (load_percent/100))
            print("Loading %d of %d gaits in memory for %s" %
                  (loadMax, len(self.gaitNames), self.is_train == True and "train" or "test"))
            gaits = []
            lengths = []
            labels = []
            loadedNum = 0
            newGaitNames = []
            for g in self.gaitNames:
                newGaitNames.append(g)
                gaits.append(DatasetEGait.gaits.get(g)[()].tolist())
                lengths.append(DatasetEGait.lengths.get(g)[()].tolist())
                labels.append(DatasetEGait.labels.get(g)[()].tolist())
                loadedNum += 1
                if loadedNum >= loadMax:
                    break

            self.gaitNames = newGaitNames
            self.gaits = np.array(gaits).astype(np.float32)
            self.lengths = np.array(lengths).astype(np.int64)
            self.labels = np.array(labels).astype(np.int64)
            print("Done loading %s gaits for %s" % (loadedNum, self.is_train == True and "train" or "test"))

    def loadDataSet(self):
        datasetFile = 'emotion-gait.h5'
        if os.path.isfile(datasetFile) or (not os.path.isfile(os.path.join(DatasetEGait.path, 'features.h5'))):
            if not os.path.isfile(datasetFile):
                DatasetEGait.downloadFile('http://nneimanis.id.lv/emotion-gait/emotion-gait.h5')
            print("Loading dataset")
            DatasetEGait.hf = h5py.File(datasetFile, 'r')
            if 'gaits' not in DatasetEGait.hf:
                print("Invalid dataset")
                sys.exit(1)
            gaits = DatasetEGait.hf['gaits']

            # Randomly split in test and train data
            print("Randomly splitting in train and test parts")
            testItemsNum = int((len(gaits) / 5))
            while len(DatasetEGait.test_gaits) < testItemsNum:
                n = random.randint(0, len(gaits) - 1)
                if n not in DatasetEGait.test_gaits:
                    DatasetEGait.test_gaits.append(n)

            for i in range(len(gaits)):
                if i not in DatasetEGait.test_gaits:
                    DatasetEGait.train_gaits.append(i)

            gaitNames = list(DatasetEGait.hf['gaits'].keys())
            test = []
            train = []
            for i in DatasetEGait.test_gaits:
                test.append(gaitNames[i])
            for i in DatasetEGait.train_gaits:
                train.append(gaitNames[i])

            DatasetEGait.train_gaits = train
            DatasetEGait.test_gaits = test

            print("Train len: %d" % len(DatasetEGait.train_gaits))
            print("Test len: %d" % len(DatasetEGait.test_gaits))
            DatasetEGait.gaits = DatasetEGait.hf['gaits']
            DatasetEGait.lengths = DatasetEGait.hf['lengths']
            DatasetEGait.labels = DatasetEGait.hf['labels']
            DatasetEGait.loaded = 1

        else:
            print("Generating emotion-gait dataset")
            skiplist_ELMD = ["000412", "000413", "000422", "000423", "000734", "001367", "001590", "001666",  # jumps
                             "000348", "000403", "000412", "000413", "001590", "001707", "001626", "001643",
                             "001666"  # no movement
                             ]

            hf = h5py.File(datasetFile, 'w')
            gaits = hf.create_group("gaits")
            lengths = hf.create_group("lengths")
            labels = hf.create_group("labels")

            for file, labelsFile in [
                ["features.h5", "labels.h5"],
                ["features_rotated-y.h5", "labels_rotated-y.h5"],
                ["features_rotated-y-scaled.h5", "labels_rotated-y-scaled.h5"],
                ["features_ELMD_centered.h5", "labels_ELMD.h5"],
                ["features_ELMD_rotated-y.h5", "labels_ELMD_rotated-y.h5"],
                ["features_ELMD_rotated-y-scaled.h5", "labels_ELMD_rotated-y-scaled.h5"]
            ]:
                print("\tLoading %s" % file)
                inputGaitFile = DatasetEGait.openFile(os.path.join(DatasetEGait.path, file))
                inputLabelFile = None
                if labelsFile:
                    inputLabelFile = DatasetEGait.openFile(os.path.join(DatasetEGait.path, labelsFile))

                for gaitName in inputGaitFile.keys():
                    if 'ELMD' in file:
                        if gaitName[:6] in skiplist_ELMD:
                            print("Skipping gait %s" % gaitName)
                            continue

                    if gaitName in gaits:
                        print("Error: gait %s already exists in dataset" % gaitName)
                        exit(1)
                    newFrames = []
                    frames = inputGaitFile[gaitName]
                    # zero pad frames so we have 240 frame for each gait
                    newFrames = newFrames + list(frames)
                    while len(newFrames) < 240:
                        newFrames.append(frames[len(frames)-1])

                    gaits.create_dataset(
                        gaitName,
                        data=np.array(newFrames).astype(np.float16),
                        compression="gzip"
                    )
                    lengths.create_dataset(
                        gaitName,
                        data=np.array(int(len(frames))).astype(np.int32),
                    )
                    labels.create_dataset(
                        gaitName,
                        data=np.array(int(inputLabelFile[gaitName][()])).astype(np.int32),
                    )

                    if np.any(newFrames > np.finfo(np.dtype(np.half)).max):
                        print("Error: Gait %s coordinate larger than %.1f" %
                              (gaitName, np.finfo(np.dtype(np.half)).min))
                        exit(1)
                    if np.any(newFrames < np.finfo(np.dtype(np.half)).min):
                        print("Error: Gait %s coordinate smaller than %.1f" %
                              (gaitName, np.finfo(np.dtype(np.half)).min))
                        exit(1)

            hf.close()
            exit(0)

    def __len__(self):
        return len(self.gaitNames)

    def __getitem__(self, idx):
        if self.in_memory:
            return self.gaits[idx], self.lengths[idx], self.labels[idx]
        return (np.array(DatasetEGait.gaits[self.gaitNames[idx]]).astype(np.float32),
                np.int64(DatasetEGait.lengths[self.gaitNames[idx]]),
                np.int64(DatasetEGait.labels[self.gaitNames[idx]]))

    def openFile(file):
        try:
            inputFile = h5py.File(file, 'r')
        except OSError as e:
            print("Failed to open file %s: %s" % (file, e.strerror))
            sys.exit(1)
        return inputFile

    def downloadFile(url):
        import requests
        local_filename = url.split('/')[-1]
        with requests.get(url, stream=True) as r:
            total_length = int(r.headers.get('content-length'))
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in tqdm(r.iter_content(chunk_size=8192), total=int(total_length/8192), unit=' chunks'):
                    f.write(chunk)
        return local_filename


data_loader_train = torch.utils.data.DataLoader(
    dataset=DatasetEGait(is_train=True, in_memory=args.dataset_no_memory, load_percent=args.load_memory_percent),
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True
)

data_loader_test = torch.utils.data.DataLoader(
    dataset=DatasetEGait(is_train=False, in_memory=args.dataset_no_memory, load_percent=args.load_memory_percent),
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=True
)
print("Dataset loaded")

class ModelRNN(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.ff_1 = torch.nn.Linear(
            in_features=3*16,
            out_features=hidden_size
        )
        self.lstm = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=args.lstm_layers,
            batch_first=True
        )
        self.ff_class = torch.nn.Linear(
            in_features=2*hidden_size,
            out_features=4
        )

    def forward(self, x, len):
        x_flat = self.ff_1.forward(x)
        cudnn_fmt = torch.nn.utils.rnn.PackedSequence(x_flat, len)
        hidden, cells = self.lstm.forward(cudnn_fmt[0])

        hidden = hidden.data

        h_mean = torch.mean(hidden, dim=1).squeeze()
        h_max = torch.amax(hidden, dim=1).squeeze()
        h_cat = torch.cat((h_max, h_mean), axis=1)
        logits = self.ff_class.forward(h_cat)
        y_prim = torch.softmax(logits, dim=1)
        return y_prim


if not torch.cuda.is_available() or args.device != 'cuda':
    args.device = 'cpu'
else:
    print("Using cuda")

model = ModelRNN(hidden_size=args.hidden_size)

if args.load_model:
    if os.path.isfile(modelSaveFile):
        print("Loaded model from %s" % modelSaveFile)
        model.load_state_dict(torch.load(modelSaveFile))
        model.eval()

model.to(args.device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.learning_rate
)

metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'acc'
    ]:
        metrics[f'{stage}_{metric}'] = []

fig = plt.figure()
bestLoss = 1e20
for epoch in range(1, args.epochs):
    currentLoss = 1e20

    for data_loader in [data_loader_train, data_loader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        if data_loader == data_loader_test:
            stage = 'test'
            model = model.eval()
        else:
            model = model.train()

        for x, leghts, y_idx in data_loader:
            x = x.to(args.device)
            y_idx = y_idx.to(args.device).squeeze()

            y_prim = model.forward(x, leghts)

            idxes = torch.arange(x.size(0)).to(args.device)
            loss = -torch.mean(torch.log(y_prim[idxes, y_idx] + 1e-8))
            y_idx_prim = torch.argmax(y_prim, dim=1)

            acc = torch.mean((y_idx == y_idx_prim) * 1.0)
            metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())
            metrics_epoch[f'{stage}_acc'].append(acc.cpu().item())

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

    plt.clf()
    plts = []
    c = 0
    for key, value in metrics.items():
        plts += plt.plot(value, f'C{c}', label=key)
        ax = plt.twinx()
        c += 1
    plt.legend(plts, [it.get_label() for it in plts])
    plt.draw()
    plt.pause(.001)

    if args.save_model:
        if currentLoss < bestLoss:
            bestLoss = currentLoss
            torch.save(model.state_dict(), modelSaveFile)
            print("Saved model to %s " % modelSaveFile)