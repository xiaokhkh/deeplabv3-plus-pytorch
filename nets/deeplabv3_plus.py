import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2

class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial
        
        model           = mobilenetv2(pretrained)
        self.features   = model.features[:-1]

        self.total_idx  = len(self.features)
        self.down_idx   = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
        
    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x 


#-----------------------------------------#
#   ASPP特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取
#-----------------------------------------#
class ASPP(nn.Module):
	def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
		super(ASPP, self).__init__()
		self.branch1 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),
		)
		self.branch2 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch3 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch4 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0,bias=True)
		self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
		self.branch5_relu = nn.ReLU(inplace=True)

		self.conv_cat = nn.Sequential(
				nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),		
		)

	def forward(self, x):
		[b, c, row, col] = x.size()
        #-----------------------------------------#
        #   一共五个分支
        #-----------------------------------------#
		conv1x1 = self.branch1(x)
		conv3x3_1 = self.branch2(x)
		conv3x3_2 = self.branch3(x)
		conv3x3_3 = self.branch4(x)
        #-----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        #-----------------------------------------#
		global_feature = torch.mean(x,2,True)
		global_feature = torch.mean(global_feature,3,True)
		global_feature = self.branch5_conv(global_feature)
		global_feature = self.branch5_bn(global_feature)
		global_feature = self.branch5_relu(global_feature)
		global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
		
        #-----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        #-----------------------------------------#
		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
		result = self.conv_cat(feature_cat)
		return result

class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16):
        super(DeepLab, self).__init__()
        if backbone=="xception":
            #----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,256]
            #   主干部分    [30,30,2048]
            #----------------------------------#
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone=="mobilenet":
            #----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,24]
            #   主干部分    [30,30,320]
            #----------------------------------#
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

        #-----------------------------------------#
        #   ASPP特征提取模块
        #   利用不同膨胀率的膨胀卷积进行特征提取
        #-----------------------------------------#
        # self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16//downsample_factor)
        self.aspp = DenseASPP(dim_in=in_channels, dim_out=256, rates=[6, 12, 18, 24])
        #----------------------------------#
        #   浅层特征边
        #----------------------------------#
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )		
        # 增加自注意力模块
        # self.self_attention = SelfAttention(65536, 8)
        self.cat_conv = nn.Sequential(
            nn.Conv2d(48+256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        #-----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        #-----------------------------------------#
        low_level_features, x = self.backbone(x)
        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)

        # 使用自注意力进行特征提取
        # x = self.self_attention(x)
        #-----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        #-----------------------------------------#
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x


class StripPooling(nn.Module):
    def __init__(self, kernel_size, stride):
        super(StripPooling, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride=stride)

    def forward(self, x):
        b, c, h, w = x.size()
        h = h - h % self.pool.kernel_size[0]
        w = w - w % self.pool.kernel_size[1]
        x = x[:, :, :h, :w]
        x = self.pool(x)
        return x
    
class DenseASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rates, bn_mom=0.1):
        super(DenseASPP, self).__init__()
        self.strip_pool = StripPooling(kernel_size=(1, 4), stride=(1, 2))  # 定义条纹池化模块
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim_in+dim_out*1, dim_out, kernel_size=3, padding=rates[0], dilation=rates[0], bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(dim_in+(dim_out*1)*2, dim_out, kernel_size=3, padding=rates[1], dilation=rates[1], bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(dim_in+(dim_out*1)*3, dim_out, kernel_size=3, padding=rates[2], dilation=rates[2], bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(dim_in+(dim_out*1)*4, dim_out, kernel_size=3, padding=rates[3], dilation=rates[3], bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out*5, dim_out, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        strip_pooled = self.strip_pool(x)  # 进行条纹池化
        conv1 = self.conv1(strip_pooled)
        conv2 = self.conv2(torch.cat([strip_pooled, conv1], 1))
        conv3 = self.conv3(torch.cat([strip_pooled, conv1, conv2], 1))
        conv4 = self.conv4(torch.cat([strip_pooled, conv1, conv2, conv3], 1))
        conv5 = self.conv5(torch.cat([strip_pooled, conv1, conv2, conv3, conv4], 1))
        conv_cat = self.conv_cat(torch.cat([conv1, conv2, conv3, conv4, conv5], 1))
        return conv_cat
    
# 使用 self attention 进行特征提取
class SelfAttention(nn.Module):
    def __init__(self, in_dim, reduction_ratio=8):
        super(SelfAttention, self).__init__()
        self.channel_in = in_dim
        self.reduction_ratio = reduction_ratio

        assert in_dim % reduction_ratio == 0, "Make sure reduction_ratio divides in_dim exactly."
        self.channel_reduced = in_dim // reduction_ratio

        # Define the layers for computing query, key, and value
        self.query = nn.Linear(in_features=self.channel_in, out_features=self.channel_reduced)
        self.key = nn.Linear(in_features=self.channel_in, out_features=self.channel_reduced)
        self.value = nn.Linear(in_features=self.channel_in, out_features=self.channel_in)

        # Define the output layer for the weighted sum of values
        self.output_layer = nn.Linear(in_features=self.channel_in, out_features=self.channel_in)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Calculate query, key, and value using linear layers
        proj_query = self.query(x).view(batch_size, -1, self.channel_reduced)
        proj_query = proj_query.permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, self.channel_reduced)
        proj_value = self.value(x).view(batch_size, -1, channels * height * width)

        # Calculate attention weights
        energy = torch.bmm(proj_query, proj_key.transpose(1, 2))
        attention_weights = F.softmax(energy, dim=-1)

        # Compute the weighted sum of values
        out = torch.bmm(attention_weights, proj_value)
        out = out.view(batch_size, channels, height, width)

        # Apply output layer and residual connection
        out = self.output_layer(out)
        out = out + x
        return out
