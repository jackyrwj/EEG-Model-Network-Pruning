import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from torcheeg.model_selection import KFoldPerSubject, train_test_split, KFold
import time
from compute_flops import print_model_param_nums, count_model_param_flops
from torcheeg1.models import CCNN
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
import logging
import random



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
    print(f"Valid Error: \n Accuracy: {(correct):.8%}%, Avg loss: {loss}")

    return correct, loss


if __name__ == "__main__":
    seed_everything(42)
    os.makedirs("./tmp_out/examples_ccnn", exist_ok=True)
    logger = logging.getLogger('examples_ccnn')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(
        './tmp_out/examples_ccnn/examples_ccnn_pruned_weight30.log')
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    dataset = DEAPDataset(io_path=f'./tmp_out/examples_ccnn/deap',
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
    k_fold = KFold( n_splits=10, split_path=f'./tmp_out/examples_ccnn/split', shuffle=True)


    os.makedirs("./tmp_out/examples_ccnn_seed", exist_ok=True)
    logger = logging.getLogger('examples_ccnn_seed')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(
        # './tmp_out/examples_ccnn_seed/examples_ccnn_pruned_weight30.log')
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
            transforms.Select('emotion'),
            transforms.Lambda(lambda x: x + 1)
        ]))
    k_fold = KFold(n_splits=10, split_path=f'./tmp_out/examples_ccnn_seed/split', shuffle=True)
    # -----------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = nn.CrossEntropyLoss()
    batch_size = 64

    test_accs = []
    test_losses = []
    param_nums = []
    param_flops = []

    # -----------------------------------------
    for i, (train_dataset, test_dataset) in enumerate(k_fold.split(dataset)):
        if i < 10:
        # if (i == 4) or (i == 6) or (i == 8):
        # if i == 0:
            checkpoint = torch.load(f'./tmp_out/examples_ccnn/pruned_model/l1norm10/pruned_model{i}.pt')
            cfg = checkpoint['cfg']
            param = checkpoint['model']
            model = CCNN(
                    num_classes=2, 
                    in_channels=4, 
                    grid_size=(9, 9),
                    conv_channel_1=cfg[0],
                    conv_channel_2=cfg[1],
                    conv_channel_3=cfg[2],
                    conv_channel_4=cfg[3]
                    ).cuda()
            model.load_state_dict(param)

            
            model_ref = CCNN(
                num_classes=2, 
                # num_classes=3, 
                in_channels=4,grid_size=(9, 9)).cuda()

            model_ref.load_state_dict(torch.load(
                f'./tmp_out/examples_ccnn/unpruned_model/model{i}.pt'))
            weight
            percent = 0.3

            model = model_ref
            total = 0
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
                    print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.
                          format(k, mask.numel(), int(torch.sum(mask))))
            print('Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}'.format(
                total, pruned, pruned/total))

            --------------------------------------------------
            # l1-norm
            origin = [64, 128, 256, 64]

            percent = 0.1   #10
            # percent = 0.3   #30 
            # percent = 0.8   #80

            cfg = [round(channel_i * (1 - percent)) for channel_i in origin]     

            cfg_mask = []
            layer_id = 0
            for m in model_ref.modules():
                if isinstance(m, nn.Conv2d):
                    out_channels = m.weight.data.shape[0]
                    if out_channels == cfg[layer_id]:
                        cfg_mask.append(torch.ones(out_channels))
                        layer_id += 1
                        continue
                    weight_copy = m.weight.data.abs().clone()
                    weight_copy = weight_copy.cpu().numpy()
                    L1_norm = np.sum(weight_copy, axis=(1, 2, 3))
                    arg_max = np.argsort(L1_norm)
                    arg_max_rev = arg_max[::-1][:cfg[layer_id]]
                    assert arg_max_rev.size == cfg[layer_id], "size of arg_max_rev not correct"
                    mask = torch.zeros(out_channels)
                    mask[arg_max_rev.tolist()] = 1
                    cfg_mask.append(mask)
                    layer_id += 1
            # print(len(cfg_mask))
            # print(cfg_mask)
            
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
            # print(model)

            start_mask = torch.ones(4)
            layer_id_in_cfg = 0
            end_mask = cfg_mask[layer_id_in_cfg]
            for [m0, m1] in zip(model_ref.modules(), model.modules()):
                if isinstance(m0, nn.Conv2d):
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    print('In shape: {:d}, Out shape {:d}.'.format(
                        idx0.size, idx1.size))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                    w1 = w1[idx1.tolist(), :, :, :].clone()
                    m1.weight.data = w1.clone()

                    start_mask = end_mask
                    layer_id_in_cfg += 1
                    if layer_id_in_cfg < len(cfg_mask):
                        end_mask = cfg_mask[layer_id_in_cfg]

                elif isinstance(m0, nn.Linear):
                    # if layer_id_in_cfg == len(cfg_mask):
                    #     idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask[-1].cpu().numpy())))
                    #     if idx0.size == 1:
                    #         idx0 = np.resize(idx0, (1,))
                    #     m1.weight.data = m0.weight.data[:, idx0].clone()
                    #     m1.bias.data = m0.bias.data.clone()
                    #     layer_id_in_cfg += 1
                    #     continue
                    # m1.weight.data = m0.weight.data.clone()
                    # m1.bias.data = m0.bias.data.clone()
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


            
            origin = [64, 128, 256, 64]

            percent = 0.9

            # cfg = [round(channel_i * (1 - percent)) for channel_i in origin]
            cfg = origin
            cfg[2] = round(cfg[2] * (1 - percent))
            print(cfg)

            cfg_mask = []
            layer_id = 0
            for m in model_ref.modules():
                if isinstance(m, nn.Conv2d):
                    out_channels = m.weight.data.shape[0]
                    if out_channels == cfg[layer_id]:
                        cfg_mask.append(torch.ones(out_channels))
                        layer_id += 1
                        continue
                    weight_copy = m.weight.data.abs().clone()
                    weight_copy = weight_copy.cpu().numpy()
                    L1_norm = np.sum(weight_copy, axis=(1, 2, 3))
                    arg_max = np.argsort(L1_norm)
                    arg_max_rev = arg_max[::-1][:cfg[layer_id]]
                    assert arg_max_rev.size == cfg[layer_id], "size of arg_max_rev not correct"
                    mask = torch.zeros(out_channels)
                    mask[arg_max_rev.tolist()] = 1
                    cfg_mask.append(mask)
                    layer_id += 1
            # print(len(cfg_mask))
            # print(cfg_mask)
            
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
            # print(model)

            start_mask = torch.ones(4)
            layer_id_in_cfg = 0
            end_mask = cfg_mask[layer_id_in_cfg]
            for [m0, m1] in zip(model_ref.modules(), model.modules()):
                if isinstance(m0, nn.Conv2d):
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    print('In shape: {:d}, Out shape {:d}.'.format(
                        idx0.size, idx1.size))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                    w1 = w1[idx1.tolist(), :, :, :].clone()
                    m1.weight.data = w1.clone()

                    start_mask = end_mask
                    layer_id_in_cfg += 1
                    if layer_id_in_cfg < len(cfg_mask):
                        end_mask = cfg_mask[layer_id_in_cfg]

                elif isinstance(m0, nn.Linear):
                    # if layer_id_in_cfg == len(cfg_mask):
                    #     idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask[-1].cpu().numpy())))
                    #     if idx0.size == 1:
                    #         idx0 = np.resize(idx0, (1,))
                    #     m1.weight.data = m0.weight.data[:, idx0].clone()
                    #     m1.bias.data = m0.bias.data.clone()
                    #     layer_id_in_cfg += 1
                    #     continue
                    # m1.weight.data = m0.weight.data.clone()
                    # m1.bias.data = m0.bias.data.clone()
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
            
            torch.save(model.state_dict(),
            model.load_state_dict(torch.load(
            f'./tmp_out/examples_ccnn/pruned_model/weight30/pruned_model{i}.pt')

            
            # l1
            torch.save({
                'cfg': cfg,
                'model': model.state_dict()}, 
            # model.load_state_dict(torch.load(
            f'./tmp_out/examples_ccnn/pruned_model/new_l1norm10/pruned_model{i}.pt')

            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
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
        f"Avg Test Error: \n Accuracy: {np.mean(test_accs):.8%}%, Avg loss: {np.mean(test_losses)}")
    logger.info(
        f"Avg size: {np.mean(param_nums)/ 1e6}M, Avg float: {np.mean(param_flops)  / 1e9}G")
