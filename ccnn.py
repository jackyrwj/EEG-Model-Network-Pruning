from typing import Tuple

import torch
import torch.nn as nn
from compute_flops import print_model_param_nums, count_model_param_flops


class CCNN(torch.nn.Module):
    r'''
    Continuous Convolutional Neural Network (CCNN). For more details, please refer to the following information.

    - Paper: Yang Y, Wu Q, Fu Y, et al. Continuous convolutional neural network with 3D input for EEG-based emotion recognition[C]//International Conference on Neural Information Processing. Springer, Cham, 2018: 433-443.
    - URL: https://link.springer.com/chapter/10.1007/978-3-030-04239-4_39
    - Related Project: https://github.com/ynulonger/DE_CNN

    Below is a recommended suite for use in emotion recognition tasks:

    .. code-block:: python

        dataset = DEAPDataset(io_path=f'./deap',
                    root_path='./data_preprocessed_python',
                    offline_transform=transforms.Compose([
                        transforms.BandDifferentialEntropy(),
                        transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
                    ]),
                    online_transform=transforms.ToTensor(),
                    label_transform=transforms.Compose([
                        transforms.Select('valence'),
                        transforms.Binary(5.0),
                    ]))
        model = CCNN(num_classes=2, in_channels=4, grid_size=(9, 9))

    Args:
        in_channels (int): The feature dimension of each electrode. (defualt: :obj:`4`)
        grid_size (tuple): Spatial dimensions of grid-like EEG representation. (defualt: :obj:`(9, 9)`)
        num_classes (int): The number of classes to predict. (defualt: :obj:`2`)
        dropout (float): Probability of an element to be zeroed in the dropout layers. (defualt: :obj:`0.25`)
    '''
    def __init__(self, in_channels: int = 4, grid_size: Tuple[int, int] = (9, 9), num_classes: int = 2, dropout: float = 0.5,
                                        conv_channel_1: int = 64,
                                        conv_channel_2: int = 128,
                                        conv_channel_3: int = 256,
                                        conv_channel_4: int = 64
                                        ):
        super().__init__()
        self.in_channels = in_channels
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.conv_channel_1 = conv_channel_1
        self.conv_channel_2 = conv_channel_2
        self.conv_channel_3 = conv_channel_3
        self.conv_channel_4 = conv_channel_4
        

        self.conv1 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)), nn.Conv2d(self.in_channels, self.conv_channel_1, kernel_size=4, stride=1),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)), nn.Conv2d(self.conv_channel_1, self.conv_channel_2, kernel_size=4, stride=1), nn.ReLU())
        self.conv3 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)), nn.Conv2d(self.conv_channel_2, self.conv_channel_3, kernel_size=4, stride=1), nn.ReLU())
        self.conv4 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)), nn.Conv2d(self.conv_channel_3, self.conv_channel_4, kernel_size=4, stride=1), nn.ReLU())

        self.lin1 = nn.Sequential(
            # nn.Linear(self.grid_size[0] * self.grid_size[1] * 64, 1024),
            nn.Linear(self.grid_size[0] * self.grid_size[1] * conv_channel_4, 1024),
            nn.SELU(), # Not mentioned in paper
            nn.Dropout2d(self.dropout)
        )
        self.lin2 = nn.Linear(1024, self.num_classes)

    @property
    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, self.in_channels, *self.grid_size)

            mock_eeg = self.conv1(mock_eeg)
            mock_eeg = self.conv2(mock_eeg)
            mock_eeg = self.conv3(mock_eeg)
            mock_eeg = self.conv4(mock_eeg)
            mock_eeg = mock_eeg.flatten(start_dim=1)

            return mock_eeg.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 4, 9, 9]`. Here, :obj:`n` corresponds to the batch size, :obj:`4` corresponds to :obj:`in_channels`, and :obj:`(9, 9)` corresponds to :obj:`grid_size`.

        Returns:
            torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
        '''
    
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.flatten(start_dim=1)
        x = self.lin1(x)
        x = self.lin2(x)
        # x = self.dequant(x)
        return x

if __name__ == '__main__':
    input = torch.randn(64,4,9,9).cuda()
    model = CCNN(num_classes=2, in_channels=4, grid_size=(9, 9)).cuda()
    print(model)

    print_model_param_nums(model).cpu()
    count_model_param_flops(model, channel=4, x=9, y=9).cpu()
