import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,3"
# os.environ['CUDA_VISIBLE_DEVICES'] = "4,5"
os.environ['CUDA_VISIBLE_DEVICES'] = "6,7"
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torcheeg import transforms
from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.constants.emotion_recognition.deap import \
    DEAP_CHANNEL_LOCATION_DICT
from torcheeg.model_selection import KFold
from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.model_selection import KFoldPerSubject
from torcheeg.model_selection import train_test_split
from torcheeg.models import TSCeption
from torch.nn.utils import prune
import logging
import time
from sklearn.metrics import accuracy_score


DEAP_CHANNEL_LIST = [
    'FP1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'OZ', 'PZ', 'FP2', 'AF4',
    'FZ', 'F4', 'F8', 'FC6', 'FC2', 'CZ', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'
]
gpus = [0, 1]
logging.basicConfig(filename='/home/raowj/Data/torcheeg/example.log',  format='%(message)s', level=logging.INFO)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# class CNN(torch.nn.Module):
#     def __init__(self, in_channels=4, num_classes=3):
#         super().__init__()
#         self.conv1 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)), nn.Conv2d(in_channels, 64, kernel_size=4, stride=1),
#                                    nn.ReLU())
#         self.conv2 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)), nn.Conv2d(64, 128, kernel_size=4, stride=1), nn.ReLU())
#         self.conv3 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)), nn.Conv2d(128, 256, kernel_size=4, stride=1), nn.ReLU())
#         self.conv4 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)), nn.Conv2d(256, 64, kernel_size=4, stride=1), nn.ReLU())

#         self.lin1 = nn.Linear(9 * 9 * 64, 1024)
#         self.lin2 = nn.Linear(1024, num_classes)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)

#         x = x.flatten(start_dim=1)
#         x = self.lin1(x)
#         x = self.lin2(x)
#         return x


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch_idx, batch in enumerate(dataloader):
        # X = batch[0].to(device)
        # y = batch[1].to(device)
        X = batch[0].cuda()
        y = batch[1].cuda()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            loss, current = loss.item(), batch_idx * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            # logging.info("loss: %f  [%d/%d]",loss,current,size)


def valid(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            # X = batch[0].to(device)
            # y = batch[1].to(device)
            X = batch[0].cuda()
            y = batch[1].cuda()

            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    val_loss /= num_batches
    correct /= size
    print(f"Valid Error: \n Accuracy: {(100*correct)}%, Avg loss: {loss}")

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)



if __name__ == "__main__":
    seed_everything(42)

    # os.makedirs("./tmp_out/examples_torch", exist_ok=True)

    # dataset = DEAPDataset(io_path=f'./tmp_out/examples_torch/deap',
    #                       root_path='./tmp_in/data_preprocessed_python',
    #                       offline_transform=transforms.Compose([
    #                           transforms.BandDifferentialEntropy(apply_to_baseline=True),
    #                           transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT, apply_to_baseline=True)
    #                       ]),
    #                       online_transform=transforms.Compose([transforms.BaselineRemoval(),
    #                                                            transforms.ToTensor()]),
    #                       label_transform=transforms.Compose([
    #                           transforms.Select('valence'),
    #                           transforms.Binary(5.0),
    #                       ]),
    #                     #   num_worker=4)
    #                       num_worker=50)
    # 5 fold
    # cv = KFold(n_splits=5, shuffle=True, split_path='/home/raowj/Data/torcheeg/tmp_out/examples_torch/split')
    # 10 fold
    cv = KFold(n_splits=10, shuffle=True, split_path='/home/raowj/Data/torcheeg/tmp_out/examples_torch/split')
    # cv = LeaveOneSubjectOut('/home/raowj/Data/torcheeg/examples/split/loso')
    # train_dataset, test_dataset = train_test_split(dataset=dataset, split_path='./split')

    dataset = DEAPDataset(io_path=f'/home/raowj/Data/torcheeg/tmp_out/examples_torch/deap',
                        root_path='/home/raowj/Data/torcheeg/tmp_in/data_preprocessed_python',
            chunk_size=512,
            baseline_num=1,
            baseline_chunk_size=512,
            offline_transform=transforms.Compose([
                transforms.PickElectrode(transforms.PickElectrode.to_index_list(
                ['FP1', 'AF3', 'F3', 'F7',
                'FC5', 'FC1', 'C3', 'T7',
                'CP5', 'CP1', 'P3', 'P7',
                'PO3','O1', 'FP2', 'AF4',
                'F4', 'F8', 'FC6', 'FC2',
                'C4', 'T8', 'CP6', 'CP2',
                'P4', 'P8', 'PO4', 'O2'], DEAP_CHANNEL_LIST)),
                transforms.To2d()
            ]),
            online_transform=transforms.ToTensor(),
            label_transform=transforms.Compose([
                transforms.Select('valence'),
                transforms.Binary(5.0),
            ]))



    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = nn.CrossEntropyLoss()
    batch_size = 64

    # for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
    for train_dataset, val_dataset in cv.split(dataset):

        # model = CNN().to(device)
        model = TSCeption(num_classes=2,
                    num_electrodes=28,
                    sampling_rate=128,
                    num_T=15,
                    num_S=15,
                    hid_channels=32,
                    dropout=0.5).cuda()
# ---------------------------------------------------
        #三个实验：不剪     剪30 剪50 剪80
        #全局剪枝
        parameters_to_prune = (
            (model.Tception1[0], 'weight'),
            (model.Tception2[0], 'weight'),
            (model.Tception3[0], 'weight'),
            (model.Sception1[0], 'weight'),
            (model.Sception2[0], 'weight'),
            (model.fusion_layer[0], 'weight'),
        )
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            # amount=0.3,
            # amount=0.5,
            amount=0.8,
        )

        # #非结构L1剪枝
        # elif(args.mode == 'L1Unstructured'):
        #     L1UnstructuredPruner = prune.L1Unstructured(amount=2)
        #     L1UnstructuredPruner.apply(model.Sception1[0],name = "weight", amount=0.5)

        model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # epochs = 50
        # epochs = 1
        epochs = 500
        # start = time.time()
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train(train_loader, model, loss_fn, optimizer)
            valid(val_loader, model, loss_fn)
        # end = time.time()
        # logging.info('spend: %f', end - start)
        print("Done!")
       
