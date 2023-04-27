import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from torcheeg.model_selection import KFoldPerSubject, train_test_split, KFold
import time
import math
from compute_flops import print_model_param_nums, count_model_param_flops
from torcheeg1.models import CCNN
import sys
from torcheeg.datasets.constants.emotion_recognition.seed import \
    SEED_CHANNEL_LOCATION_DICT
from torcheeg.datasets.constants.emotion_recognition.deap import \
    DEAP_CHANNEL_LOCATION_DICT
from torcheeg.datasets import DEAPDataset, SEEDDataset
from torcheeg import transforms
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch
import numpy as np
import logging

import random
from torch.utils.tensorboard import SummaryWriter



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
        X = batch[0].to(device)
        y = batch[1].to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()


        # sft & weight
        for k, m in enumerate(model.modules()):
            if isinstance(m, nn.Conv2d):
                weight_copy = m.weight.data.abs().clone()
                mask = weight_copy.gt(0).float().cuda()
                m.weight.grad.data.mul_(mask)

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
        for batch in dataloader:
            X = batch[0].to(device)
            y = batch[1].to(device)

            pred = model(X)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    loss /= num_batches
    correct /= size
    print(f"Valid Error: \n Accuracy: {(100*correct)}%, Avg loss: {loss}")

    return correct, loss


if __name__ == "__main__":
    seed_everything(42)
    os.makedirs("./tmp_out/examples_ccnn", exist_ok=True)
    logger = logging.getLogger('examples_ccnn')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(
    './tmp_out/examples_ccnn/examples_ccnn_scratch_weight30.log')
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    dataset = DEAPDataset(io_path=f'./tmp_out/examples_ccnn/deap',
                          root_path='./tmp_in/data_preprocessed_python',
                          offline_transform=transforms.Compose([
                              transforms.BandDifferentialEntropy(
                                  apply_to_baseline=True),
                              transforms.BaselineRemoval(),
                              transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT),
                              transforms.MeanStdNormalize(),
                          ]),
                          online_transform=transforms.ToTensor(),
                          label_transform=transforms.Compose([
                              transforms.Select('valence'),
                              transforms.Binary(5.0),
                          ]),
                          num_worker=8)
    k_fold = KFold(n_splits=10, split_path=f'./tmp_out/examples_ccnn/split', shuffle=True)
    # k_fold = KFoldPerSubject(n_splits=10, split_path=f'./tmp_out/examples_ccnn/split_KFPS', shuffle=True)


    os.makedirs("./tmp_out/examples_ccnn_seed", exist_ok=True)
    logger = logging.getLogger('examples_ccnn_seed')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(
    # './tmp_out/examples_ccnn_seed/examples_ccnn_scratch_weight30.log')
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    dataset = SEEDDataset(
        io_path=f'./tmp_out/examples_ccnn_seed/seed',
        root_path='./tmp_in/Preprocessed_EEG',
        offline_transform=transforms.Compose([
            transforms.BandDifferentialEntropy(),
            transforms.ToGrid(SEED_CHANNEL_LOCATION_DICT)
        ]),
        online_transform=transforms.ToTensor(),
        label_transform=transforms.Compose([
            transforms.Select(['emotion']),
            transforms.Lambda(lambda x: x[0] + 1)
        ]))
    k_fold = KFold(n_splits=10, split_path=f'./tmp_out/examples_ccnn_seed/split', shuffle=True)
    # -----------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = nn.CrossEntropyLoss()
    batch_size = 64

    test_accs = []
    test_losses = []
    param_nums = []
    param_flops = []
    # epochs = 1
    epochs = 200
    logger.info(f"epoch:{epochs}")
    # writer = SummaryWriter()
    # -----------------------------------------------
    for i, (train_dataset, test_dataset) in enumerate(k_fold.split(dataset)):
        # if 0 <= i <= 4:
        # if 5 <= i <= 9:
        # if i < 10:
        # if (1 <= i <= 4) or (6 <= i <= 9):
        # if i == 0:
        if i >= 0:
        
            # checkpoint
            checkpoint = torch.load(
                f'./tmp_out/examples_ccnn/pruned_model/weight30/pruned_model{i}.pt')

            # weight
            # cfg = [64,128,256,64]

            # l1-norm
            # cfg = checkpoint['cfg']

            # sft
            cfg = [64,128,256,64]

            model = CCNN(
                num_classes=2,
                # num_classes=3,
                in_channels=4, 
                grid_size=(9, 9),
                conv_channel_1=cfg[0],
                conv_channel_2=cfg[1],
                conv_channel_3=cfg[2],
                conv_channel_4=cfg[3]
                ).to(device)
            model_ref = CCNN(
                num_classes=2,
                # num_classes=3,
                in_channels=4, 
                grid_size=(9, 9),
                conv_channel_1=cfg[0],
                conv_channel_2=cfg[1],
                conv_channel_3=cfg[2],
                conv_channel_4=cfg[3]
                ).to(device)

                
            # weight
            # model_ref.load_state_dict(checkpoint)

            # sft
            model_ref.load_state_dict(checkpoint)


            # weight & sft
            for m, m_ref in zip(model.modules(), model_ref.modules()):
                if isinstance(m, nn.Conv2d):
                    weight_copy = m_ref.weight.data.abs().clone()
                    mask = weight_copy.gt(0).float().cuda()
                    n = mask.sum() / float(m.in_channels)
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    m.weight.data.mul_(mask)

            # --------------------------------------------------------
            # optimizer=torch.optim.Adam(model.parameters(), lr=1e-4)
            learning_rate = 0.01
            optimizer=torch.optim.SGD(model.parameters(), lr = learning_rate)
            train_dataset, val_dataset = train_test_split(train_dataset,
                                                        test_size=0.2,
                                                        split_path=f'./tmp_out/examples_ccnn/split{i}',
                                                        # split_path=f'./tmp_out/examples_ccnn_seed/split{i}',
                                                        # split_path=f'./tmp_out/examples_ccnn/split{i}_KFPS',
                                                        # split_path=f'./tmp_out/examples_ccnn_seed/split{i}_KFPS',
                                                        shuffle=True)
            # 手动cuda                                               
            t1 = time.time()
            train_dataset_cuda = []
            val_dataset_cuda = []
            for x in range(len(train_dataset)):
                train_dataset_cuda.append((train_dataset[x][0].cuda(), train_dataset[x][1]))
            for y in range(len(val_dataset)):
                val_dataset_cuda.append((val_dataset[y][0].cuda(), val_dataset[y][1]))
            t2 = time.time()
            print(t2 - t1)

            

            train_loader=DataLoader(
                train_dataset_cuda, batch_size=batch_size, shuffle=True)
                # train_dataset, batch_size=batch_size, shuffle=True)
            val_loader=DataLoader(
                val_dataset_cuda, batch_size=batch_size, shuffle=True)
                # val_dataset, batch_size=batch_size, shuffle=True)
            best_val_acc=0.0


            lr_decay_epochs = [150,180]
            lr_decay_rate = 0.1
            for t in range(epochs):
                t3 = time.time()
                # 修改epochs
                steps = np.sum(t > np.asarray(lr_decay_epochs))
                if steps > 0:
                    new_lr = learning_rate * (lr_decay_rate ** steps)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                
                print(f'Fold:[{i}/9]    Epoch:[{t}/{epochs}]:')
                train_loss = train(train_loader, model, loss_fn, optimizer)
                # writer.add_scalars(
                    f"ccnn/scratch/weight30/model{i}/loss", {"Train":train_loss}, t)

                val_acc, val_loss=valid(val_loader, model, loss_fn)
                writer.add_scalars(
                    f"ccnn/scratch/weight30/model{i}/loss", {"Val":val_loss}, t)


                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(
                    ), f'./tmp_out/examples_ccnn/scratch_model/weight30/scratch_model{i}.pt')
                t4 = time.time()
                print(t4 - t3)
                    
            model.load_state_dict(torch.load(
                f'./tmp_out/examples_ccnn/scratch_model/weight30/scratch_model{i}.pt'))

            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False)
            test_acc, test_loss = valid(test_loader, model, loss_fn)
            logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            logger.info(
                f"Test Error {i}: \n Accuracy: {(100*test_acc)}%, Avg loss: {test_loss}")

            test_accs.append(test_acc)
            test_losses.append(test_loss)
            param_nums.append(print_model_param_nums(model).cpu())
            param_flops.append(count_model_param_flops(model, channel=4, x=9, y=9).cpu())
            print()

    logger.info(
        f"Avg Test Error: \n Accuracy: {100*np.mean(test_accs)}%, Avg loss: {np.mean(test_losses)}")
    logger.info(
        f"Avg size: {np.mean(param_nums)/ 1e6}M, Avg float: {np.mean(param_flops)  / 1e9}G")
