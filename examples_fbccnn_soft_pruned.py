import os
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
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
        # './tmp_out/examples_fbccnn/examples_fbccnn_fine-tuned_soft70.log')
        './tmp_out/examples_fbccnn/examples_fbccnn_fine-tuned_soft80.log')
        # './tmp_out/examples_fbccnn/examples_fbccnn_fine-tuned_soft90.log')
        # './tmp_out/examples_fbccnn/examples_fbccnn_fine-tuned_soft20.log')

        # './tmp_out/examples_fbccnn/examples_fbccnn_fine-tuned_soft70_KFPS.log')
        # './tmp_out/examples_fbccnn/examples_fbccnn_fine-tuned_soft90_KFPS.log')
        # './tmp_out/examples_fbccnn/examples_fbccnn_fine-tuned_soft20_KFPS.log')
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
    # k_fold = KFoldPerSubject(n_splits=10, split_path=f'./tmp_out/examples_fbccnn/split_KFPS', shuffle=True)

    # os.makedirs("./tmp_out/examples_fbccnn_seed", exist_ok=True)
    # logger = logging.getLogger('examples_fbccnn_seed')
    # logger.setLevel(logging.DEBUG)
    # console_handler = logging.StreamHandler()
    # file_handler = logging.FileHandler(
        # './tmp_out/examples_fbccnn_seed/examples_fbccnn_fine-tuned_soft70.log')
    # logger.addHandler(console_handler)
    # logger.addHandler(file_handler)
    # dataset = SEEDDataset(
    #     io_path=f'./tmp_out/examples_fbccnn_seed/seed',
    #     root_path='./tmp_in/Preprocessed_EEG',
    #     offline_transform=transforms.Compose([
    #         transforms.BandDifferentialEntropy(),
    #         transforms.ToGrid(SEED_CHANNEL_LOCATION_DICT)
    #     ]),
    #     online_transform=transforms.ToTensor(),
    #     label_transform=transforms.Compose([
    #         transforms.Select('emotion'),
    #         transforms.Lambda(lambda x: x + 1)
    #     ]))
    # k_fold = KFold(n_splits=10, split_path=f'./tmp_out/examples_fbccnn_seed/split', shuffle=True)
    # k_fold = KFoldPerSubject(n_splits=10, split_path=f'./tmp_out/examples_fbccnn_seed/split_KFPS', shuffle=True)
    # -----------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = nn.CrossEntropyLoss()
    batch_size = 64

    test_accs = []
    test_losses = []
    param_nums = []
    param_flops = []
    epochs = 200
    # logger.info(f"epoch:{epochs}")
    # writer = SummaryWriter()
    # -----------------------------------------
    for i, (train_dataset, test_dataset) in enumerate(k_fold.split(dataset)):
        # if i < 10:
        # if i < 10:
        # if 0 <= i <= 4:
        # if 5 <= i <= 9:
        # if i >= 0:
        # if i == 0:
            model = FBCCNN(
                num_classes=2, 
                # num_classes=3, 
                in_channels=4, 
                grid_size=(9, 9)
            ).cuda()
            model.load_state_dict(torch.load(
                f'./tmp_out/examples_fbccnn/unpruned_model/model{i}.pt'))
                # f'./tmp_out/examples_fbccnn_seed/unpruned_model/model{i}.pt'))
                # f'./tmp_out/examples_fbccnn/unpruned_model/model{i}_KFPS.pt'))
                # f'./tmp_out/examples_fbccnn_seed/unpruned_model/model{i}_KFPS.pt'))
                

            learning_rate = 0.01
            optimizer=torch.optim.SGD(model.parameters(), lr = learning_rate)
            # optimizer=torch.optim.Adam(model.parameters(), lr=1e-4)
            train_dataset, val_dataset = train_test_split(train_dataset,
                                                        test_size=0.2,
                                                        split_path=f'./tmp_out/examples_fbccnn/split{i}',
                                                        # split_path=f'./tmp_out/examples_fbccnn_seed/split{i}',

                                                        # split_path=f'./tmp_out/examples_fbccnn/split{i}_KFPS',
                                                        # split_path=f'./tmp_out/examples_fbccnn_seed/split{i}_KFPS',
                                                        shuffle=True)
            # # 手动cuda                                               
            # t1 = time.time()
            # train_dataset_cuda = []
            # val_dataset_cuda = []
            # for x in range(len(train_dataset)):
            #     train_dataset_cuda.append((train_dataset[x][0].cuda(), train_dataset[x][1]))
            # for y in range(len(val_dataset)):
            #     val_dataset_cuda.append((val_dataset[y][0].cuda(), val_dataset[y][1]))
            # t2 = time.time()
            # print(t2 - t1)

            

            train_loader=DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True)
                # train_dataset_cuda, batch_size=batch_size, shuffle=True)
            val_loader=DataLoader(
                val_dataset, batch_size=batch_size, shuffle=True)
                # val_dataset_cuda, batch_size=batch_size, shuffle=True)
            best_val_acc = 0.0
            # ----------------------------------------
            # percent = 0.7
            # percent = 0.8
            # percent = 0.9
            percent = 0.2
            # ----------------------------------------
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
                # print(f'Fold:[{i}/31]    Epoch:[{t}/{epochs}]:')
                train_loss = train(train_loader, model, loss_fn, optimizer)
            
                if (t % 1 == 0 or t == epochs-1):
                    # init_length
                    model_size = {}
                    model_length = {}
                    compress_rate = {}
                    mat = {}
                    for index, item in enumerate(model.parameters()):
                        # model_size : {0 : [64,4,4,4], 1:[64]...}
                        model_size[index] = item.size()

                    # model_lenght : {0:4096, 1:64 ....}
                    for index1 in model_size:
                        for index2 in range(0, len(model_size[index1])):
                            if index2 == 0:
                                model_length[index1] = model_size[index1][0]
                            else:
                                model_length[index1] *= model_size[index1][index2]

                    # init_rate
                    for index, item in enumerate(model.parameters()):
                        compress_rate[index] = 1
                    # compress_rate([0.7, 1, 1, 0.7.....])
                    for key in range(0, 25, 4):
                        compress_rate[key] = percent
                    # mask_index(0, 3, 6, 9....)
                    mask_index = [x for x in range(0, 25, 4)]

                    # init_mask
                    for index, item in enumerate(model.parameters()):
                        if(index in mask_index):
                            # get_filter_codebook
                            codebook = np.ones(model_length[index])
                            if len(item.data.size()) == 4:
                                filter_pruned_num = int(item.data.size()[0]*(1-compress_rate[index]))
                                weight_vec = item.data.view(item.data.size()[0], -1)
                                norm2 = torch.norm(weight_vec, 2, 1)
                                norm2_np = norm2.cpu().numpy()
                                filter_index = norm2_np.argsort()[:filter_pruned_num]
                                kernel_length = item.data.size(
                                )[1] * item.data.size()[2] * item.data.size()[3]
                                for x in range(0, len(filter_index)):codebook[filter_index[x] * kernel_length: (filter_index[x]+1) * kernel_length] = 0
                                # print("filter codebook done")
                            mat[index] = torch.FloatTensor(codebook)
                            mat[index] = mat[index].cuda()
                    # print("mask Ready")
                    # do_mask
                    for index, item in enumerate(model.parameters()):
                        if(index in mask_index):
                            a = item.data.view(model_length[index])
                            b = a * mat[index]
                            item.data = b.view(model_size[index])
                    print("mask Done")

                    val_acc, val_loss = valid(val_loader, model, loss_fn)
                    # writer.add_scalar(
                        # f"fbccnn/sft70/model{i}", val_acc, t)
                        # f"fbccnn/sft80/model{i}", val_acc, t)
                        # f"fbccnn/sft90/model{i}", val_acc, t)
                        # f"fbccnn/sft20/model{i}", train_loss, t)
                        # f"fbccnn/seed/sft20/model{i}", train_loss, t)
                    # if val_acc > best_val_acc:
                    #     best_val_acc = val_acc
                    #     torch.save(model.state_dict(
                        # ), f'./tmp_out/examples_fbccnn/fine-tuned_model/sft70/fine-tuned_model{i}.pt')
                # t4 = time.time()
                # print(t4 - t3)

            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            # model.load_state_dict(torch.load(
                # f'./tmp_out/examples_fbccnn/fine-tuned_model/sft70/fine-tuned_model{i}.pt'))

            logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            test_acc, test_loss = valid(test_loader, model, loss_fn)
            logger.info(
                f"Test Error {i}: \n Accuracy: {(100*test_acc)}%, Avg loss: {test_loss}")

            test_accs.append(test_acc)
            test_losses.append(test_loss)
            param_nums.append(print_model_param_nums(model).cpu())
            param_flops.append(count_model_param_flops(
                model, channel=4, x=9, y=9).cpu())
            print()

    logger.info(
        f"Avg Test Error: \n Accuracy: {100*np.mean(test_accs)}%, Avg loss: {np.mean(test_losses)}")
    logger.info(
        f"Avg size: {np.mean(param_nums)/ 1e6}M, Avg float: {np.mean(param_flops)  / 1e9}G")
