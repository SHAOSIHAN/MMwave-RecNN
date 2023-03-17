import torch
import torch.nn as nn
import torch.nn.functional as F

class Baseline(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, numInput=402):
        super(Baseline, self).__init__()

        self.dense = nn.Linear(in_features=numInput, out_features=400)

        self.conv2dT1 = nn.ConvTranspose2d(in_channels=100, out_channels=50,
                                           kernel_size=3, stride=2,
                                           padding=1, output_padding=1)

        self.conv2dT2 = nn.ConvTranspose2d(in_channels=50, out_channels=25,
                                           kernel_size=3, stride=2,
                                           padding=1, output_padding=1)

        self.conv2dT3 = nn.ConvTranspose2d(in_channels=25, out_channels=12,
                                           kernel_size=3, stride=2,
                                           padding=1, output_padding=1)

        self.conv2dT4 = nn.ConvTranspose2d(in_channels=12, out_channels=6,
                                           kernel_size=3, stride=2,
                                           padding=1, output_padding=1)

        self.conv2dT5 = nn.ConvTranspose2d(in_channels=6, out_channels=3,
                                           kernel_size=3, stride=2,
                                           padding=1, output_padding=1)

        self.conv2dT6 = nn.ConvTranspose2d(in_channels=3, out_channels=2,
                                           kernel_size=3, stride=2,
                                           padding=1, output_padding=1)

        self.conv2dT7 = nn.ConvTranspose2d(in_channels=2, out_channels=out_ch,
                                           kernel_size=3, stride=2,
                                           padding=1, output_padding=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.dense(x))  # shape
        #x = self.drop1(x)
        x = torch.reshape(x, (-1, 100, 2, 2))

        x = F.relu(self.conv2dT1(x))
        x = F.relu(self.conv2dT2(x))
        x = F.relu(self.conv2dT3(x))
        x = F.relu(self.conv2dT4(x))
        x = F.relu(self.conv2dT5(x))
        x = F.relu(self.conv2dT6(x))
        x = F.relu(self.conv2dT7(x))

        out = self.sigmoid(x)

        return out


if __name__ == '__main__':
    from torchsummary import summary
    model = Baseline(in_ch=1, out_ch=1)
    a = torch.randn(1, 1, 1, 402)
    out = model(a)
    print(out)
    summary(model, input_size=(1, 1, 402), batch_size=32)











