import logging
import os

from sklearn.model_selection import KFold
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# os.environ['CUDA_VISIBLE_DEVICES'] = "2"
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
# os.environ['CUDA_VISIBLE_DEVICES'] = "4"
# os.environ['CUDA_VISIBLE_DEVICES'] = "5"
# os.environ['CUDA_VISIBLE_DEVICES'] = "6"
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
import random
import pretty_errors

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torcheeg import transforms
from torcheeg.datasets import DEAPDataset,SEEDDataset
from torcheeg.datasets.constants.emotion_recognition.deap import \
    DEAP_CHANNEL_LOCATION_DICT
from torcheeg.datasets.constants.emotion_recognition.seed import \
    SEED_CHANNEL_LOCATION_DICT
import sys
sys.path.append("..")
from torcheeg.model_selection import KFoldPerSubject, train_test_split, KFoldGroupbyTrial, LeaveOneSubjectOut
from torcheeg.model_selection.k_fold import KFold

from torcheeg1.models import FBCCNN
# from compute_flops import print_model_param_nums, count_model_param_flops
from flops import print_model_param_nums, count_model_param_flops
import time
import examples
from loss_v2 import grad_cam_loss_v2

threshold = 0.01
# threshold = 0.09
sr = 0.00001
# def updateBN():
#     for m in model.modules():
#         if isinstance(m, nn.BatchNorm2d):
#             m.weight.grad.data.add_(0.0001 * torch.sign(m.weight.data))  # L1

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

        # loss = loss_fn(pred, y)
        loss = 10.0 * grad_cam_loss_v2(model, X, y, mode = 'ALL_FRONT', layer = 'block2')

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        model.update_skeleton(sr, threshold)
        # updateBN()
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
    os.makedirs("./tmp_out/examples_fbccnn", exist_ok=True)
    logger = logging.getLogger('examples_fbccnn')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    # file_handler = logging.FileHandler('./tmp_out/examples_fbccnn/fbccnn_pruned_swp_DEAP.log')

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # dataset = DEAPDataset(
    #                 io_path=f'./tmp_out/examples_fbccnn/deap_tmp',
    #                 root_path='./tmp_in/data_preprocessed_python',
    #                     offline_transform=transforms.Compose([
    #                         transforms.BandDifferentialEntropy(
    #                             # sampling_rate=128, 
    #                             apply_to_baseline=True),
    #                         transforms.BaselineRemoval(),
    #                         transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
    #                     ]),
    #                     online_transform=transforms.ToTensor(),
    #                     label_transform=transforms.Compose([
    #                         transforms.Select('valence'),
    #                         transforms.Binary(5.0),
    #                     ]),
    #                     chunk_size=128,
    #                     baseline_chunk_size=128,
    #                     # num_baseline=3,
    #                     num_worker=20,
    #                     )

    # dataset = DEAPDataset(
    #             io_path=f'./tmp_out/examples_fbccnn/deap',
    #             root_path='./tmp_in/data_preprocessed_python',
    #             offline_transform=transforms.Compose([
    #                 transforms.BandDifferentialEntropy(apply_to_baseline=True),
    #                 transforms.BaselineRemoval(), 
    #                 # transforms.MeanStdNormalize(),
    #                 transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
    #             ]),
    #             online_transform=transforms.ToTensor(),
    #             label_transform=transforms.Compose([
    #                 transforms.Select('valence'),
    #                 transforms.Binary(5.0),
    #             ]),
    #             num_worker=20
    #             )
    # k_fold = KFold(n_splits=10, split_path=f'./tmp_out/examples_fbccnn/split', shuffle=True)


    dataset = SEEDDataset(
            io_path=f'./tmp_out/examples_fbccnn_seed/seed',
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
    k_fold = KFold(n_splits=10, split_path=f'./tmp_out/examples_fbccnn_seed/split', shuffle=True)



    #     os.makedirs("./tmp_out/examples_fbccnn_seed", exist_ok=True)
    #     logger = logging.getLogger('examples_fbccnn_seed')
    #     logger.setLevel(logging.DEBUG)
    #     console_handler = logging.StreamHandler()
    #     # file_handler = logging.FileHandler('./tmp_out/examples_fbccnn_seed/examples_fbccnn_unpruned_seed.log')
    #     file_handler = logging.FileHandler('./tmp_out/examples_fbccnn_seed/examples_fbccnn_unpruned_seed_updateBN.log')
    #     # file_handler = logging.FileHandler('./tmp_out/examples_fbccnn_seed/examples_fbccnn_unpruned_seed_KFPS.log')
    #     logger.addHandler(console_handler)
    #     logger.addHandler(file_handler)
    #     dataset = SEEDDataset(
    #         io_path=f'./tmp_out/examples_fbccnn_seed/seed',
    #         root_path='./tmp_in/Preprocessed_EEG',
    #         offline_transform=transforms.Compose([
    #             transforms.BandDifferentialEntropy(),
    #             transforms.ToGrid(SEED_CHANNEL_LOCATION_DICT)
    #         ]),
    #         online_transform=transforms.ToTensor(),
    #         label_transform=transforms.Compose([
    #             transforms.Select('emotion'),
    #             transforms.Lambda(lambda x: x + 1)
    #         ]))
    #     k_fold = KFold(n_splits=10, split_path=f'./tmp_out/examples_fbccnn_seed/split', shuffle=True)
    #     # k_fold = KFoldPerSubject(n_splits=10, split_path=f'./tmp_out/examples_fbccnn_seed/split_KFPS', shuffle=True)

    # --------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = nn.CrossEntropyLoss()
    batch_size = 64

    test_accs = []
    test_losses = []
    param_nums = []
    param_flops = []
    epochs = 100
    # epochs = 1
    # logger.info(f"epochs:{epochs}")
    # writer = SummaryWriter()
    # -----------------------------------------------
    for i, (train_dataset, test_dataset) in enumerate(k_fold.split(dataset)):
        if 3 <= i <= 6:
        # if i >= 0:
        # if i == 0:
            # model = FBCCNN(
            #     num_classes=2, 
            #     # num_classes=3, 
            #     in_channels=4, grid_size=(9, 9)).to(device)
            # model.load_state_dict(torch.load(
            #     f'./tmp_out/examples_fbccnn/unpruned_model/model{i}.pt'))

            # model = examples.__dict__['VGG'](num_classes=2).cuda()
            model = examples.__dict__['VGG'](num_classes=3).cuda()
            # model.load_state_dict(torch.load(f'./tmp_out/examples_fbccnn/pruned_model/swp/model{i}.pt'))
            # model.prune(threshold)


            # ----------------------------------------------------
            # learning_rate = 1e-4
            # optimizer=torch.optim.Adam(model.parameters(), lr = learning_rate) # official: weight_decay=5e-1
            learning_rate = 1e-2
            optimizer=torch.optim.SGD(model.parameters(), lr = learning_rate) 
            train_dataset, val_dataset = train_test_split(train_dataset,
                                                                test_size=0.2,
                                                                # split_path=f'./tmp_out/examples_fbccnn/split{i}',
                                                                split_path=f'./tmp_out/examples_fbccnn_seed/split{i}',
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
            

            # lr_decay_epochs = [150, 180]
            # lr_decay_rate = 0.1
            for t in range(epochs):
                # if:
                #     # t3 = time.time()
                #     # 修改epochs
                #     # steps = np.sum(t > np.asarray(lr_decay_epochs))
                #     # if steps > 0:
                #     #     new_lr = learning_rate * (lr_decay_rate ** steps)
                #     #     for param_group in optimizer.param_groups:
                #     #         param_group['lr'] = new_lr

                
                print(f'Fold:[{i}/9]    Epoch:[{t}/{epochs}]:')
                train_loss = train(train_loader, model, loss_fn, optimizer)
                val_acc, val_loss=valid(val_loader, model, loss_fn)
                if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        torch.save(model.state_dict(), 
                        # f'./tmp_out/examples_fbccnn/pruned_model/fbccnn_pruned_swp_DEAP/model{i}.pt')
                # t4 = time.time()
                # print(t4 - t3)
            

            model.load_state_dict(torch.load(
                # f'./tmp_out/examples_fbccnn/pruned_model/fbccnn_pruned_swp_DEAP/model{i}.pt'))

            model.prune(threshold)

            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            test_acc, test_loss = valid(test_loader, model, loss_fn)
            logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            logger.info(f"Test Error {i}: \n Accuracy: {(100*test_acc)}%, Avg loss: {test_loss}")

            test_accs.append(test_acc)
            test_losses.append(test_loss)
            # param_nums.append(print_model_param_nums(model).cpu())
            # param_flops.append(count_model_param_flops(model,channel = 4, x = 9, y = 9).cpu())
            param_nums.append(print_model_param_nums(model))
            param_flops.append(count_model_param_flops(model,channel = 4, x = 9, y = 9))
            print()

    logger.info(f"Avg Test Error: \n Accuracy: {np.mean(test_accs):.2%}, Avg loss: {np.mean(test_losses)}")
    logger.info(f"Avg Params: {np.mean(param_nums)/ 1e6}M, Avg FLOPs: {np.mean(param_flops) / 1e8}")


    