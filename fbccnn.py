from typing import Tuple

import torch
import torch.nn as nn
from compute_flops import print_model_param_nums, count_model_param_flops


class FBCCNN(nn.Module):
    r'''
    Frequency Band Correlation Convolutional Neural Network (FBCCNN). For more details, please refer to the following information.

    - Paper: Pan B, Zheng W. Emotion Recognition Based on EEG Using Generative Adversarial Nets and Convolutional Neural Network[J]. Computational and Mathematical Methods in Medicine, 2021.
    - URL: https://www.hindawi.com/journals/cmmm/2021/2520394/

    Below is a recommended suite for use in emotion recognition tasks:

    .. code-block:: python

        dataset = DEAPDataset(io_path=f'./deap',
                    root_path='./data_preprocessed_python',
                    online_transform=transforms.Compose([
                        transforms.BandPowerSpectralDensity(),
                        transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
                    ]),
                    label_transform=transforms.Compose([
                        transforms.Select('valence'),
                        transforms.Binary(5.0),
                    ]))
        model = FBCCNN(num_classes=2, in_channels=4, grid_size=(9, 9))

    Args:
        in_channels (int): The feature dimension of each electrode, i.e., :math:`N` in the paper. (defualt: :obj:`4`)
        grid_size (tuple): Spatial dimensions of grid-like EEG representation. (defualt: :obj:`(9, 9)`)
        num_classes (int): The number of classes to predict. (defualt: :obj:`2`)
    '''
    def __init__(self, in_channels: int = 4, grid_size: Tuple[int, int] = (9, 9), num_classes: int = 2,
                                conv_channel_1: int = 12,
                                conv_channel_2: int = 32,
                                conv_channel_3: int = 64,
                                conv_channel_4: int = 128,
                                conv_channel_5: int = 256,
                                conv_channel_6: int = 128,
                                conv_channel_7: int = 32,
                                ):
        super(FBCCNN, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.grid_size = grid_size
        self.conv_channel_1 = conv_channel_1
        self.conv_channel_2 = conv_channel_2
        self.conv_channel_3 = conv_channel_3
        self.conv_channel_4 = conv_channel_4
        self.conv_channel_5 = conv_channel_5
        self.conv_channel_6 = conv_channel_6
        self.conv_channel_7 = conv_channel_7
        # ---------------------
        # self.quant = torch.quantization.QuantStub()
        # self.dequant = torch.quantization.DeQuantStub()
        # ----------------------


        # self.block1 = nn.Sequential(nn.Conv2d(in_channels, 12, kernel_size=3, padding=1, stride=1), nn.ReLU(),
        #                             nn.BatchNorm2d(12))
        # self.block2 = nn.Sequential(nn.Conv2d(12, 32, kernel_size=3, padding=1, stride=1), nn.ReLU(),
        #                             nn.BatchNorm2d(32))
        # self.block3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1), nn.ReLU(),
        #                             nn.BatchNorm2d(64))
        # self.block4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1), nn.ReLU(),
        #                             nn.BatchNorm2d(128))
        # self.block5 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1), nn.ReLU(),
        #                             nn.BatchNorm2d(256))
        # self.block6 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1, stride=1), nn.ReLU(),
        #                             nn.BatchNorm2d(128))
        # self.block7 = nn.Sequential(nn.Conv2d(128, 32, kernel_size=3, padding=1, stride=1), nn.ReLU(),
        #                             nn.BatchNorm2d(32))
        self.block1 = nn.Sequential(nn.Conv2d(in_channels, self.conv_channel_1, kernel_size=3, padding=1, stride=1), nn.ReLU(),
                                    nn.BatchNorm2d(self.conv_channel_1))
        self.block2 = nn.Sequential(nn.Conv2d(self.conv_channel_1, self.conv_channel_2, kernel_size=3, padding=1, stride=1), nn.ReLU(),
                                    nn.BatchNorm2d(self.conv_channel_2))
        self.block3 = nn.Sequential(nn.Conv2d(self.conv_channel_2, self.conv_channel_3, kernel_size=3, padding=1, stride=1), nn.ReLU(),
                                    nn.BatchNorm2d(self.conv_channel_3))
        self.block4 = nn.Sequential(nn.Conv2d(self.conv_channel_3, self.conv_channel_4, kernel_size=3, padding=1, stride=1), nn.ReLU(),
                                    nn.BatchNorm2d(self.conv_channel_4))
        self.block5 = nn.Sequential(nn.Conv2d(self.conv_channel_4, self.conv_channel_5, kernel_size=3, padding=1, stride=1), nn.ReLU(),
                                    nn.BatchNorm2d(self.conv_channel_5))
        self.block6 = nn.Sequential(nn.Conv2d(self.conv_channel_5, self.conv_channel_6, kernel_size=3, padding=1, stride=1), nn.ReLU(),
                                    nn.BatchNorm2d(self.conv_channel_6))
        self.block7 = nn.Sequential(nn.Conv2d(self.conv_channel_6, self.conv_channel_7, kernel_size=3, padding=1, stride=1), nn.ReLU(),
                                    nn.BatchNorm2d(self.conv_channel_7))
        # self.lin1 = nn.Sequential(nn.Linear(grid_size[0] * grid_size[1] * 32, 512), nn.ReLU())
        self.lin1 = nn.Sequential(nn.Linear(grid_size[0] * grid_size[1] * conv_channel_7, 512), nn.ReLU())
        self.lin2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU())
        self.lin3 = nn.Linear(128, num_classes)

    @property
    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, self.in_channels, *self.grid_size)

            mock_eeg = self.block1(mock_eeg)
            mock_eeg = self.block2(mock_eeg)
            mock_eeg = self.block3(mock_eeg)
            mock_eeg = self.block4(mock_eeg)
            mock_eeg = self.block5(mock_eeg)
            mock_eeg = self.block6(mock_eeg)
            mock_eeg = self.block7(mock_eeg)

            mock_eeg = mock_eeg.flatten(start_dim=1)

            return mock_eeg.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 4, 9, 9]`. Here, :obj:`n` corresponds to the batch size, :obj:`4` corresponds to :obj:`in_channels`, and :obj:`(9, 9)` corresponds to :obj:`grid_size`.

        Returns:
            torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
        '''
        # 量化-----------
        # x = self.quant(x)
        # -------------
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        x = x.flatten(start_dim=1)

        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        # 量化-----------
        # x = self.dequant(x)
        # -------------
        return x

if __name__ == '__main__':
    input = torch.randn(64,4,9,9).cuda()
    model = FBCCNN(num_classes=2, in_channels=4, grid_size=(9, 9)).cuda()
    # print(model)

    
    print_model_param_nums(model).cpu()
    count_model_param_flops(model, channel=4, x=9, y=9).cpu()
    print('aaa')
    # output = model(input)
    # print(output.size())