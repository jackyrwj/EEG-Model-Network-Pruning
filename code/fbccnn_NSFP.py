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
        './tmp_out/examples_fbccnn/examples_fbccnn_pruned_slimming30.log')

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
                in_channels=4,
                grid_size=(9, 9),).cuda()
            model_ref.load_state_dict(torch.load(
                f'./tmp_out/examples_fbccnn/unpruned_model/model{i}.pt'))
                # f'./tmp_out/examples_fbccnn_seed/unpruned_model/model{i}.pt'))

    
            # network-sliming
            total = 0
            for m in model_ref.modules():
                if isinstance(m, nn.BatchNorm2d):
                    total += m.weight.data.shape[0]

            bn = torch.zeros(total)
            index = 0
            for m in model_ref.modules():
                if isinstance(m, nn.BatchNorm2d):
                    size = m.weight.data.shape[0]
                    bn[index:(index+size)] = m.weight.data.abs().clone()
                    index += size
            y, b = torch.sort(bn)

            # percent = 0.1
            percent = 0.3
            # percent = 0.8

            thre_index = int(total * percent)
            thre = y[thre_index]
            pruned = 0
            cfg = []
            cfg_mask = []
            for k, m in enumerate(model_ref.modules()):
                if isinstance(m, nn.BatchNorm2d):
                    weight_copy = m.weight.data.abs().clone()
                    # mask = weight_copy.gt(thre).float().cuda()
                    mask = weight_copy.ge(thre).float().cuda()
                    # pruned = pruned + mask.shape[0] - torch.sum(mask)
                    m.weight.data.mul_(mask)
                    m.bias.data.mul_(mask)
                    # mask(0,1,0,0,0,1...)
                    cfg.append(int(torch.sum(mask)))
                    cfg_mask.append(mask.clone())
                    print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                        format(k, mask.shape[0], int(torch.sum(mask))))
            print(cfg)
            pruned_ratio = pruned/total
            # print(pruned_ratio)
            print('Pre-processing Successful!')

            model = FBCCNN(
                num_classes=2, 
                # num_classes=3, 
                in_channels=4, 
                grid_size=(9, 9),
                conv_channel_1=cfg[0],
                conv_channel_2=cfg[1],
                conv_channel_3=cfg[2],
                conv_channel_4=cfg[3],
                conv_channel_5=cfg[4],
                conv_channel_6=cfg[5],
                conv_channel_7=cfg[6],
                ).to(device)
            # print(model)

            layer_id_in_cfg = 0
            start_mask = torch.ones(4)
            end_mask = cfg_mask[layer_id_in_cfg]
            for [m0, m1] in zip(model_ref.modules(), model.modules()):
                if isinstance(m0, nn.BatchNorm2d):
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                    m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                    m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                    m1.running_var = m0.running_var[idx1.tolist()].clone()

                    layer_id_in_cfg += 1
                    start_mask = end_mask.clone()
                    if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                        end_mask = cfg_mask[layer_id_in_cfg]

                elif isinstance(m0, nn.Conv2d):
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                    w1 = w1[idx1.tolist(), :, :, :].clone()
                    m1.weight.data = w1.clone()


                elif isinstance(m0, nn.Linear):
                    if layer_id_in_cfg == len(cfg_mask):
                        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                        if idx0.size == 1:
                            idx0 = np.resize(idx0, (1,))
                        tmp = torch.zeros(len(start_mask), 9, 9)
                        tmp[idx0, :, :] = 1
                        tmp = tmp.flatten().int()
                        idx0 = np.squeeze(np.argwhere(tmp))
                        m1.weight.data = m0.weight.data[:, idx0.tolist()].clone()
                        m1.bias.data = m0.bias.data.clone()   
                        layer_id_in_cfg += 1   
                    else:
                        m1.weight.data = m0.weight.data.clone()
                        m1.bias.data = m0.bias.data.clone()

            
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
