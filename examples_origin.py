from ast import arg
import logging
import os

from sklearn.model_selection import KFold
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torcheeg import transforms
from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.constants.emotion_recognition.deap import \
    DEAP_CHANNEL_LOCATION_DICT,DEAP_CHANNEL_LIST
import sys
sys.path.append("..")
from torcheeg.model_selection import KFoldPerSubject, train_test_split, KFoldGroupbyTrial
from torcheeg.model_selection.k_fold import KFold

from torcheeg1.models import TSCeption
from compute_flops import print_model_param_nums, count_model_param_flops
import argparse


parser = argparse.ArgumentParser()
# Datasets
parser.add_argument('-d', '--dataset', default='deap', type=str)
# parser.add_argument('-d', '--dataset', default='seed', type=str)
# parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
#                     help='number of data loading workers (default: 4)')

# Optimization options
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--percent', default=0.6, type=float)
parser.add_argument('--folds', default=10, type=int)

# Architecture/Model
parser.add_argument('--model', type=str, default='tsception', help='Model')
# parser.add_argument('--model_selection', type=str, default='KFoldGroupbyTrial', help='Model selection')
parser.add_argument('--model_selection', type=str, default='KFold', help='Model selection')

# dir
parser.add_argument('--save_log', default='./tmp_out/examples_tsception/examples_tsception_unpruned.log', type=str)
parser.add_argument('--save_dir', default='./tmp_out/examples_tsception/unpruned_model/', type=str)

#Device options
# parser.add_argument('--gpu', default='7', type=str,
#                     help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch_idx, batch in enumerate(dataloader):
        for  batch in dataloader:
            if args.model == 'fbccnn':
                X = batch[0].float().to(device)
            else:
                X = batch[0].to(device)
            y = batch[1].to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            loss, current = loss.item(), batch_idx * len(X)
            print(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return loss


def valid(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    loss, correct = 0, 0
    with torch.no_grad():
        for  batch in dataloader:
            if args.model == 'fbccnn':
                X = batch[0].float().to(device)
            else:
                X = batch[0].to(device)
            y = batch[1].to(device)

            pred = model(X)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    loss /= num_batches
    correct /= size
    print(f"Valid Error: \n Accuracy: {(correct):.8%}%, Avg loss: {loss}")

    return correct, loss




if __name__ == "__main__":
    seed_everything(42)
    os.makedirs("./tmp_out/examples_tsception", exist_ok=True)

    logger = logging.getLogger('examples_tsception')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(args.save_log)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


    if args.dataset == 'deap':
        dataset_dir = 'data_preprocessed_python'
    elif args.dataset == 'seed':
        dataset_dir = 'Preprocessed_EEG'
    else:
        print('false')
    dataset = DEAPDataset(
        io_path=f'./tmp_out/examples_tsception/{args.dataset}',
        root_path= f'./tmp_in/{dataset_dir}',
        chunk_size=512,
        baseline_num=1,
        baseline_chunk_size=512,
        offline_transform=transforms.Compose([
            transforms.PickElectrode(
                transforms.PickElectrode.to_index_list([
                    'FP1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'FP2',
                    'AF4', 'F4', 'F8', 'FC6', 'FC2', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'
                ], DEAP_CHANNEL_LIST)),
            transforms.To2d()
        ]),
        online_transform=transforms.ToTensor(),
        label_transform=transforms.Compose([
            transforms.Select('valence'),
            transforms.Binary(5.0),
        ]))

    if args.model_selection == 'KFold':
        k_fold = KFold(n_splits= args.folds, split_path=f'./tmp_out/examples_tsception/split', shuffle=True)
    elif args.model_selection == 'KFoldGroupbyTrial':
        k_fold = KFoldGroupbyTrial(n_splits= args.folds, split_path=f'./tmp_out/examples_tsception/split', shuffle=True)
    else:
        print('false')

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = nn.CrossEntropyLoss()
    batch_size = 64

    test_accs = []
    test_losses = []
    param_nums = []
    param_flops = []

    # ------------------------训练测试-----------------------
    for fold, (train_dataset, test_dataset) in enumerate(k_fold.split(dataset)):
        # 加载模型
        model = TSCeption(num_classes=2,
                        num_electrodes=28,
                        sampling_rate=128,
                        num_T=15,
                        num_S=15,
                        hid_channels=32,
                        dropout=0.5).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # official: weight_decay=5e-1
        # 划分训练/验证
        train_dataset, val_dataset = train_test_split(train_dataset,
                                                            test_size=0.2,
                                                            split_path=f'./tmp_out/examples_tsception/split{fold}',
                                                            shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


        best_val_acc = 0.0
        for epoch in range(args.epochs):
            print(f'Fold:[{fold}/9]    Epoch:[{epoch}/{args.epochs}]:')
            # -------------------------训练验证-------------------------------  
            train_loss = train(train_loader, model, loss_fn, optimizer)
            val_acc, val_loss = valid(val_loader, model, loss_fn)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # torch.save(model.state_dict(), f'./tmp_out/examples_tsception/unpruned_model/model{fold}.pt')
                torch.save(model.state_dict(), args.save_dir + f'model{fold}.pt')
        # -----------------测试----------------------------
        # model.load_state_dict(torch.load(f'./tmp_out/examples_tsception/unpruned_model/model{fold}.pt'))
        model.load_state_dict(torch.load(args.save_dir + f'model{fold}.pt'))
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        test_acc, test_loss = valid(test_loader, model, loss_fn)
        logger.info(f"Test Error {fold}: \n Accuracy: {(test_acc):.8%}%, Avg loss: {test_loss}")

        test_accs.append(test_acc)
        test_losses.append(test_loss)
        param_nums.append(print_model_param_nums(model).cpu())
        if args.model == 'ccnn' or args.model == 'fbccnn':
            param_flops.append(count_model_param_flops(model,channel = 4, x = 9, y = 9).cpu())
        elif args.model == 'tsception':
            param_flops.append(count_model_param_flops(model,channel = 1, x = 28, y = 512).cpu())
        print()



    logger.info(f"Avg Test Error: \n Accuracy: {np.mean(test_accs):.8%}%, Avg loss: {np.mean(test_losses)}")
    logger.info(f"Avg size: {np.mean(param_nums)/ 1e6}M, Avg float: {np.mean(param_flops)  / 1e9}G")

    