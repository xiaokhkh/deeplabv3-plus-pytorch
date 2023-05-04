import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2

class DenseASPP(nn.Module):
    def __init__(self, in_channel=320, out_channel=256, kernel_sizes=[3, 6, 9, 12]):
        super(DenseASPP, self).__init__()
        self.conv0 = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        self.conv1 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)

        self.dense_block1 = nn.Sequential(
            nn.Conv2d(in_channel + out_channel, out_channel, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dense_block2 = nn.Sequential(
            nn.Conv2d(in_channel + out_channel * 2, out_channel, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dense_block3 = nn.Sequential(
            nn.Conv2d(in_channel + out_channel * 3, out_channel, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dense_block4 = nn.Sequential(
            nn.Conv2d(in_channel + out_channel * 4, out_channel, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        dilated_convs = []
        for kernel_size in [3, 6, 9, 12]:
            dilated_convs.append(F.relu(self.conv1(F.pad(conv2, [(kernel_size // 2, kernel_size // 2) * 2]))))

        dilated_convs = torch.cat(dilated_convs, dim=1)
        dense_block1 = self.dense_block1(torch.cat([conv0, conv1, dilated_convs], dim=1))
        dense_block2 = self.dense_block2(torch.cat([conv0, conv1, dilated_convs, dense_block1], dim=1))
        dense_block3 = self.dense_block3(torch.cat([conv0, conv1, dilated_convs, dense_block1, dense_block2], dim=1))
        dense_block4 = self.dense_block4(torch.cat([conv0, conv1, dilated_convs, dense_block1, dense_block2, dense_block3], dim=1))

        feat = torch.cat([dilated_convs, dense_block1, dense_block2, dense_block3, dense_block4], dim=1)
        # 因为条纹池化模块不是基于卷积实现的，需要特殊处理
        # 这里略过条纹池化模块
        feat = self.conv1(F.relu(feat))
        feat = self.conv2(F.relu(feat))
        return feat


class Deeplabv3Plus(nn.Module):
    def __init__(self, in_channel=3, out_channel=1):
        super(Deeplabv3Plus, self).__init__()
        self.mobilenetv2 = mobilenet_v2(pretrained=True).features
        self.aspp = DenseASPP()
        self.self_attention = SelfAttention(in_channels=256)
        self.pred_layer = nn.Sequential(
            nn.Conv2d(640, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channel, kernel_size=1)
        )
        # 对输入图像进行3x3卷积，调整通道数，得到浅层特征
        self.ppm_conv = nn.Conv2d(96, 256, kernel_size=3, padding=1)

    def forward(self, x):
        low_level_features = self.mobilenetv2[:12](x)
        x = self.mobilenetv2[12:](x)
        x = self.daspp(x)
        x = self.self_attention(x)
        x = F.interpolate(x, scale_factor=4, mode='nearest')
        
        # 将处理后的浅层特征输入到自注意机制中
        low_level_features = self.ppm_conv(low_level_features)
        low_level_features = self.self_attention(low_level_features)
        x = torch.cat([x, low_level_features], dim=1)
        x = self.pred_layer(x)
        x = F.interpolate(x, scale_factor=4, mode='nearest')
        return x