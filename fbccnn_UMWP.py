import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import time
from compute_flops import print_model_param_nums, count_model_param_flops
from torcheeg1.models import FBCCNN
from torcheeg.model_selection.k_fold import KFold
from torcheeg.model_selection import KFoldPerSubject, train_test_split
import sys
from torcheeg.datasets.constants.emotion_recognition.deap import \
    DEAP_CHANNEL_LOCATION_DICT
from torcheeg.datasets.constants.emotion_recognition.seed import \
    SEED_CHANNEL_LOCATION_DICT
from torcheeg.datasets import DEAPDataset,SEEDDataset
from torcheeg import transforms
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch
import numpy as np
import random
import logging



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
    os.makedirs("./tmp_out/examples_fbccnn", exist_ok=True)
    logger = logging.getLogger('examples_fbccnn')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(
        './tmp_out/examples_fbccnn/examples_fbccnn_pruned_weight30.log')
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    dataset = DEAPDataset(io_path=f'./tmp_out/examples_fbccnn/deap',
                          root_path='./tmp_in/data_preprocessed_python',
                          offline_transform=transforms.Compose([
                              transforms.BandDifferentialEntropy(
                                  apply_to_baseline=True),
                              transforms.BaselineRemoval(),
                              transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
                          ]),
                          online_transform=transforms.ToTensor(),
                          label_transform=transforms.Compose([
                              transforms.Select('valence'),
                              transforms.Binary(5.0),
                          ]),
                          num_worker=8)
    k_fold = KFold(n_splits=10, split_path=f'./tmp_out/examples_fbccnn/split', shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = nn.CrossEntropyLoss()
    batch_size = 64

    test_accs = []
    test_losses = []
    param_nums = []
    param_flops = []

    for i, (train_dataset, test_dataset) in enumerate(k_fold.split(dataset)):
            model_ref = FBCCNN(
                num_classes=2, 
                # num_classes=3, 
                in_channels=4,
                grid_size=(9, 9),).cuda()
            model_ref.load_state_dict(torch.load(
                f'./tmp_out/examples_fbccnn/unpruned_model/model{i}.pt'))
                # f'./tmp_out/examples_fbccnn_seed/unpruned_model/model{i}.pt'))

            # ----------------------
            # weight
            
            # percent = 0.1
            percent = 0.3
            # percent = 0.8
            total = 0
            model = model_ref
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    total += m.weight.data.numel()
            conv_weights = torch.zeros(total)
            index = 0
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    size = m.weight.data.numel()
                    conv_weights[index:(index+size)] = m.weight.data.view(-1).abs().clone()
                    index += size

            y, b = torch.sort(conv_weights)
            thre_index = int(total * percent)
            thre = y[thre_index]
            pruned = 0
            print('Pruning threshold: {}'.format(thre))
            zero_flag = False
            for k, m in enumerate(model.modules()):
                if isinstance(m, nn.Conv2d):
                    weight_copy = m.weight.data.abs().clone()
                    mask = weight_copy.gt(thre).float().cuda()
                    pruned = pruned + mask.numel() - torch.sum(mask)
                    m.weight.data.mul_(mask)
                    if int(torch.sum(mask)) == 0:
                        zero_flag = True
                    print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.format(k, mask.numel(), int(torch.sum(mask))))
            print('Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}'.format(total, pruned, pruned/total))

            
            if zero_flag:
                print("There exists a layer with 0 parameters left.")
           
            
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            test_acc, test_loss = valid(test_loader, model, loss_fn)
            logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            logger.info(f"Test Error {i}: \n Accuracy: {(100*test_acc)}%, Avg loss: {test_loss}")

            test_accs.append(test_acc)
            test_losses.append(test_loss)
            param_nums.append(print_model_param_nums(model).cpu())
            param_flops.append(count_model_param_flops(model, channel=4, x=9, y=9).cpu())
            print()

    logger.info(f"Avg Test Error: \n Accuracy: {100*np.mean(test_accs)}%, Avg loss: {np.mean(test_losses)}")
    logger.info(f"Unprune avg size: {np.mean(param_nums)/ 1e6}M, Unprune avg float: {np.mean(param_flops)  / 1e9}G")
