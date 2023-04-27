import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from torcheeg.datasets import DEAPDataset, SEEDDataset
from sklearn.model_selection import KFold
import logging
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torcheeg import transforms
from torcheeg.datasets.constants.emotion_recognition.seed import \
    SEED_CHANNEL_LIST
from torcheeg.datasets.constants.emotion_recognition.deap import \
    DEAP_CHANNEL_LIST, DEAP_CHANNEL_LOCATION_DICT
from torcheeg.model_selection import KFoldPerSubjectGroupbyTrial, train_test_split, KFoldGroupbyTrial
from torcheeg.model_selection.k_fold_per_subject_groupby_trial import \
    KFoldPerSubjectGroupbyTrial
from torcheeg.model_selection import KFold
from torcheeg1.models import TSCeption, CCNN, FBCCNN
from compute_flops import print_model_param_nums, count_model_param_flops
import time
import argparse
import sys
import pretty_errors



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
    os.makedirs("./tmp_out/examples_tsception", exist_ok=True)
    logger = logging.getLogger('examples_tsception')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(
        # './tmp_out/examples_tsception/examples_tsception_pruned_weight30.log')
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    dataset = DEAPDataset(
        io_path=f'./tmp_out/examples_tsception/deap',
        root_path='./tmp_in/data_preprocessed_python',
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
    k_fold = KFoldGroupbyTrial(n_splits=10, split_path=f'./tmp_out/examples_tsception/split', shuffle=True)


    # os.makedirs("./tmp_out/examples_tsception_seed", exist_ok=True)
    # logger = logging.getLogger('examples_tsception_seed')
    # logger.setLevel(logging.DEBUG)
    # console_handler = logging.StreamHandler()
    # file_handler = logging.FileHandler(
        # './tmp_out/examples_tsception_seed/examples_tsception_pruned_weight30.log')
    # logger.addHandler(console_handler)
    # logger.addHandler(file_handler)
    # dataset = SEEDDataset(
    #     io_path=f'./tmp_out/examples_tsception_seed/seed',
    #     root_path='./tmp_in/Preprocessed_EEG',
    #     offline_transform=transforms.Compose([
    #         transforms.BaselineRemoval(),
    #         transforms.PickElectrode(
    #             transforms.PickElectrode.to_index_list([
    #                 'FP1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'FP2',
    #                 'AF4', 'F4', 'F8', 'FC6', 'FC2', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'
    #             ], SEED_CHANNEL_LIST)),
    #         transforms.MeanStdNormalize(),
    #     ]),
    #     online_transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.To2d(),
    #     ]),
    #     label_transform=transforms.Compose([
    #         transforms.Select('emotion'),
    #         transforms.Lambda(lambda x: x + 1)
    #     ]),
    #     num_worker=4)
    # k_fold = KFoldGroupbyTrial(n_splits= 10, split_path=f'./tmp_out/examples_tsception_seed/split', shuffle=True)

    # -------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = nn.CrossEntropyLoss()
    batch_size = 64

    test_accs = []
    test_losses = []
    param_nums = []
    param_flops = []
    # -------------------------------------------

    for i, (train_dataset, test_dataset) in enumerate(k_fold.split(dataset)):
        # if i == 0:
        # if i == 9:
        if i < 10:
        # if i == 7:
        # if 8 <= i <= 9:
        # if i >= 5:
        # if i == 2:
            model_ref = TSCeption(
                num_classes=2,
                # num_classes=3,
                num_electrodes=28,
                sampling_rate=128,
                num_T=15,
                num_S=15,
                hid_channels=32,
                dropout=0.5).to(device)
            model_ref.load_state_dict(torch.load(
                f'./tmp_out/examples_tsception/unpruned_model/model{i}.pt'))
                # f'./tmp_out/examples_tsception_seed/unpruned_model/model{i}.pt'))
                # f'./tmp_out/examples_tsception/unpruned_model/model{i}_updateBN.pt'))
                # f'./tmp_out/examples_tsception_seed/unpruned_model/model{i}_updateBN.pt'))
            # model = model_ref

                
                
            # checkpoint = torch.load(
                # f'./tmp_out/examples_tsception/pruned_model/weight30/pruned_model{i}.pt')


            # cfg = checkpoint['cfg']

            # model = TSCeption(
            #     num_classes=2,
            #     # num_classes=3,
            #     num_electrodes=28,
            #     sampling_rate=128,
            #     num_T=cfg[0],
            #     num_S=cfg[1],
            #     num_fc_norm=cfg[2],
            #     hid_channels=32,
            #     dropout=0.5
            # ).to(device)

            # # slimming
            # model.load_state_dict(checkpoint['model'])


            # --------------------------------------------
            # weight only
            # percent = 0.3
            # percent = 0.8
            # percent = 0.95
            # percent = 0.1
            # model = model_ref
            # total = 0
            # for m in model.modules():
            #     if isinstance(m, nn.Conv2d):
            #         total += m.weight.data.numel()
            # conv_weights = torch.zeros(total)
            # index = 0
            # for m in model.modules():
            #     if isinstance(m, nn.Conv2d):
            #         size = m.weight.data.numel()
            #         conv_weights[index:(index + size)] = m.weight.data.view(-1).abs().clone()
            #         index += size

            # y, b = torch.sort(conv_weights)
            # thre_index = int(total * percent)
            # thre = y[thre_index]
            # pruned = 0
            # print('Pruning threshold: {}'.format(thre))
            # zero_flag = False
            # for k, m in enumerate(model.modules()):
            #     if isinstance(m, nn.Conv2d):
            #         weight_copy = m.weight.data.abs().clone()
            #         mask = weight_copy.gt(thre).float().cuda()
            #         pruned = pruned + mask.numel() - torch.sum(mask)
            #         m.weight.data.mul_(mask)
            #         if int(torch.sum(mask)) == 0:
            #             zero_flag = True
            #         print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.format(k, mask.numel(), int(torch.sum(mask))))
            # print('Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}'.format(total, pruned, pruned/total))
            # # if zero_flag:
            # #     print("There exists a layer with 0 parameters left.")

            # # # --------------------------------------------
            # # l1-norm-pruning
            # origin = [14, 14, 14,     14,      15, 15,   15,    15,   15]
            # # origin = [15, 15, 15,     15,      14, 14,   14,    15,   15]


            # percent = 0.1   #10
            # # percent = 0.3   #30 
            # # percent = 0.8   #80

            # cfg = origin
            # # cfg = [round(channel_i * (1 - percent)) for channel_i in origin]    
            # # cfg = [int(channel_i * (1 - percent)) for channel_i in origin]    
            # print(cfg)
            # cfg_mask = []
            # layer_id = 0
            # for m in model_ref.modules():
            #     if isinstance(m, nn.Conv2d):
            #         out_channels = m.weight.data.shape[0]
            #         if out_channels == cfg[layer_id]:
            #             cfg_mask.append(torch.ones(out_channels))
            #             layer_id += 1
            #             continue
            #         weight_copy = m.weight.data.abs().clone()
            #         weight_copy = weight_copy.cpu().numpy()
            #         L1_norm = np.sum(weight_copy, axis=(1, 2, 3))
            #         arg_max = np.argsort(L1_norm)
            #         arg_max_rev = arg_max[::-1][:cfg[layer_id]]
            #         assert arg_max_rev.size == cfg[layer_id], "size of arg_max_rev not correct"
            #         mask = torch.zeros(out_channels)
            #         mask[arg_max_rev.tolist()] = 1
            #         cfg_mask.append(mask)
            #         layer_id += 1
                


            #     if isinstance(m, nn.BatchNorm2d):
            #         out_channels = m.weight.data.shape[0]
            #         if out_channels == cfg[layer_id]:
            #             cfg_mask.append(torch.ones(out_channels))
            #             layer_id += 1
            #             continue
            #         weight_copy = m.weight.data.abs().clone()
            #         weight_copy = weight_copy.cpu().numpy()
            #         # L1_norm = np.sum(weight_copy, axis=(1, 2, 3))
            #         L1_norm = weight_copy
            #         arg_max = np.argsort(L1_norm)
            #         arg_max_rev = arg_max[::-1][:cfg[layer_id]]
            #         assert arg_max_rev.size == cfg[layer_id], "size of arg_max_rev not correct"
            #         mask = torch.zeros(out_channels)
            #         mask[arg_max_rev.tolist()] = 1
            #         cfg_mask.append(mask)
            #         layer_id += 1

            # print(len(cfg_mask))
            # # print(cfg_mask)

            # model = TSCeption(
            #     num_classes=2,
            #     # num_classes=3,
            #     num_electrodes=28,
            #     sampling_rate=128,
            #     # num_T=cfg[0],
            #     # num_S=cfg[3],
            #     # num_fc_norm=cfg[5],

            #     num_T=cfg[3],
            #     num_S=cfg[6],
            #     num_fc_norm=cfg[8],

            #     hid_channels=32,
            #     dropout=0.5).to(device)
            # print(model)
            # start_mask = torch.ones(1)
            # layer_id_in_cfg = 0
            # end_mask = cfg_mask[layer_id_in_cfg]
            # index = 0
            # for [m0, m1] in zip(model_ref.modules(), model.modules()):
            #     if isinstance(m0, nn.BatchNorm2d):
            #         idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            #         if idx1.size == 1:
            #             idx1 = np.resize(idx1, (1,))
            #         m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            #         m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            #         m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            #         m1.running_var = m0.running_var[idx1.tolist()].clone()


            #         start_mask = end_mask
            #         layer_id_in_cfg += 1
            #         if layer_id_in_cfg < len(cfg_mask):
            #             end_mask = cfg_mask[layer_id_in_cfg]


            #     elif isinstance(m0, nn.Conv2d):
            #         idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            #         idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            #         # print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            #         if idx0.size == 1:
            #             idx0 = np.resize(idx0, (1,))
            #         if idx1.size == 1:
            #             idx1 = np.resize(idx1, (1,))
            #         w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            #         w1 = w1[idx1.tolist(), :, :, :].clone()
            #         m1.weight.data = w1.clone()
            #         m1.bias.data = m0.bias.data[idx1.tolist()].clone()

            #         if (index == 2) or (index == 6) or (index == 15):
            #             layer_id_in_cfg += 1
            #             if layer_id_in_cfg < len(cfg_mask):
            #                 end_mask = cfg_mask[layer_id_in_cfg]

            #         elif (index == 10) or (index == 19) or (index == 24):
            #             start_mask = end_mask
            #             layer_id_in_cfg += 1
            #             if layer_id_in_cfg < len(cfg_mask):
            #                 end_mask = cfg_mask[layer_id_in_cfg]

            #     elif isinstance(m0, nn.Linear):
            #         if layer_id_in_cfg == len(cfg_mask):
            #             idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask[-1].cpu().numpy())))
            #             if idx0.size == 1:
            #                 idx0 = np.resize(idx0, (1,))
            #             m1.weight.data = m0.weight.data[:, idx0].clone()
            #             m1.bias.data = m0.bias.data.clone()
            #             layer_id_in_cfg += 1
            #             continue
            #         m1.weight.data = m0.weight.data.clone()
            #         m1.bias.data = m0.bias.data.clone()
            #     index += 1

            # --------------------------------------------
            # # l1-norm_layer
            # origin = [15, 15, 15,     15,      15, 15,   15,    15,   15]

            # # percent = 0.1
            
            # cfg = origin
            # # cfg[0] = round(cfg[0] * (1 - percent))

            # cfg_mask = []
            # layer_id = 0
            # for m in model_ref.modules():
            #     if isinstance(m, nn.Conv2d):
            #         out_channels = m.weight.data.shape[0]
            #         if out_channels == cfg[layer_id]:
            #             cfg_mask.append(torch.ones(out_channels))
            #             layer_id += 1
            #             continue
            #         weight_copy = m.weight.data.abs().clone()
            #         weight_copy = weight_copy.cpu().numpy()
            #         L1_norm = np.sum(weight_copy, axis=(1, 2, 3))
            #         arg_max = np.argsort(L1_norm)
            #         arg_max_rev = arg_max[::-1][:cfg[layer_id]]
            #         assert arg_max_rev.size == cfg[layer_id], "size of arg_max_rev not correct"
            #         mask = torch.zeros(out_channels)
            #         mask[arg_max_rev.tolist()] = 1
            #         cfg_mask.append(mask)
            #         layer_id += 1
                


            #     if isinstance(m, nn.BatchNorm2d):
            #         out_channels = m.weight.data.shape[0]
            #         if out_channels == cfg[layer_id]:
            #             cfg_mask.append(torch.ones(out_channels))
            #             layer_id += 1
            #             continue
            #         weight_copy = m.weight.data.abs().clone()
            #         weight_copy = weight_copy.cpu().numpy()
            #         # L1_norm = np.sum(weight_copy, axis=(1, 2, 3))
            #         L1_norm = weight_copy
            #         arg_max = np.argsort(L1_norm)
            #         arg_max_rev = arg_max[::-1][:cfg[layer_id]]
            #         assert arg_max_rev.size == cfg[layer_id], "size of arg_max_rev not correct"
            #         mask = torch.zeros(out_channels)
            #         mask[arg_max_rev.tolist()] = 1
            #         cfg_mask.append(mask)
            #         layer_id += 1

            # # print(len(cfg_mask))
            # # print(cfg_mask)

            # model = TSCeption(
            #     num_classes=2,
            #     # num_classes=3,
            #     num_electrodes=28,
            #     sampling_rate=128,
            #     num_T=cfg[0],
            #     num_S=cfg[4],
            #     num_fc_norm=cfg[7],
            #     hid_channels=32,
            #     dropout=0.5).to(device)
            # # print(model)
            # start_mask = torch.ones(1)
            # layer_id_in_cfg = 0
            # end_mask = cfg_mask[layer_id_in_cfg]
            # index = 0
            # for [m0, m1] in zip(model_ref.modules(), model.modules()):
            #     if isinstance(m0, nn.BatchNorm2d):
            #         idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            #         if idx1.size == 1:
            #             idx1 = np.resize(idx1, (1,))
            #         m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            #         m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            #         m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            #         m1.running_var = m0.running_var[idx1.tolist()].clone()


            #         start_mask = end_mask
            #         layer_id_in_cfg += 1
            #         if layer_id_in_cfg < len(cfg_mask):
            #             end_mask = cfg_mask[layer_id_in_cfg]


            #     elif isinstance(m0, nn.Conv2d):
            #         idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            #         idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            #         print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            #         if idx0.size == 1:
            #             idx0 = np.resize(idx0, (1,))
            #         if idx1.size == 1:
            #             idx1 = np.resize(idx1, (1,))
            #         w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            #         w1 = w1[idx1.tolist(), :, :, :].clone()
            #         m1.weight.data = w1.clone()

            #         if (index == 2) or (index == 6) or (index == 15):
            #             layer_id_in_cfg += 1
            #             if layer_id_in_cfg < len(cfg_mask):
            #                 end_mask = cfg_mask[layer_id_in_cfg]

            #         elif (index == 10) or (index == 19) or (index == 24):
            #             start_mask = end_mask
            #             layer_id_in_cfg += 1
            #             if layer_id_in_cfg < len(cfg_mask):
            #                 end_mask = cfg_mask[layer_id_in_cfg]

            #     elif isinstance(m0, nn.Linear):
            #         if layer_id_in_cfg == len(cfg_mask):
            #             idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask[-1].cpu().numpy())))
            #             if idx0.size == 1:
            #                 idx0 = np.resize(idx0, (1,))
            #             m1.weight.data = m0.weight.data[:, idx0].clone()
            #             m1.bias.data = m0.bias.data.clone()
            #             layer_id_in_cfg += 1
            #             continue
            #         m1.weight.data = m0.weight.data.clone()
            #         m1.bias.data = m0.bias.data.clone()
            #     index += 1

            # ---------------------------------------------
            # # network-sliming
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


            percent = 0.1


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
                    if int(torch.sum(mask)) == 0:
                        mask[torch.argmax(weight_copy)] = 1
                    m.weight.data.mul_(mask)
                    m.bias.data.mul_(mask)
                    # mask(0,1,0,0,0,1...)
                    cfg.append(int(torch.sum(mask)))
                    cfg_mask.append(mask.clone())
                    print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.format(k, mask.shape[0], int(torch.sum(mask))))

            # print(cfg)
            # pruned_ratio = pruned/total
            # print(pruned_ratio)
            # print('Pre-processing Successful!')

            model = TSCeption(
                    # num_classes=2,
                    num_classes=3,
                    num_electrodes=28,
                    sampling_rate=128,
                    num_T=cfg[0],
                    num_S=cfg[1],
                    num_fc_norm=cfg[2],
                    hid_channels=32,
                    dropout=0.5).to(device)
            # print(model)

            layer_id_in_cfg = 0
            start_mask = torch.ones(1)
            end_mask = cfg_mask[layer_id_in_cfg]
            index = 0
            for [m0, m1] in zip(model_ref.modules(), model.modules()):
                if isinstance(m0, nn.BatchNorm2d):
                    idx1 = np.squeeze(np.argwhere(
                        np.asarray(start_mask.cpu().numpy())))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                    m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                    m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                    m1.running_var = m0.running_var[idx1.tolist()].clone()
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

                    if index == 10 or index == 19 or index == 24:
                        start_mask = end_mask
                        layer_id_in_cfg += 1
                        if layer_id_in_cfg < len(cfg_mask):
                            end_mask = cfg_mask[layer_id_in_cfg]

                elif isinstance(m0, nn.Linear):
                    # if index == 29:
                    if layer_id_in_cfg == len(cfg_mask):
                        idx0 = np.squeeze(np.argwhere(
                            np.asarray(cfg_mask[-1].cpu().numpy())))
                        if idx0.size == 1:
                            idx0 = np.resize(idx0, (1,))
                        m1.weight.data = m0.weight.data[:, idx0].clone()
                        m1.bias.data = m0.bias.data.clone()
                        layer_id_in_cfg += 1
                        continue
                    m1.weight.data = m0.weight.data.clone()
                    m1.bias.data = m0.bias.data.clone()
                index += 1

            # --------------测试acc-----------------
            # torch.save(model.state_dict(),
            # model.load_state_dict(torch.load(
            # f'./tmp_out/examples_tsception/pruned_model/weight30/pruned_model{i}.pt')

            


            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            test_acc, test_loss=valid(test_loader, model, loss_fn)
            logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            logger.info(
                f"Test Error {i}: \n Accuracy: {(100*test_acc)}%, Avg loss: {test_loss}")

            test_accs.append(test_acc)
            test_losses.append(test_loss)
            param_nums.append(print_model_param_nums(model).cpu())
            param_flops.append(count_model_param_flops(model, channel=1, x=28, y=512).cpu())
            print()

    logger.info(f"Avg Test Error: \n Accuracy: {np.mean(test_accs):.8%}, Avg loss: {np.mean(test_losses)}")
    logger.info(f"Avg size: {np.mean(param_nums)/ 1e6}M, Avg float: {np.mean(param_flops)  / 1e9}G\n")
