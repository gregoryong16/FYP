import torch
import torch.nn as nn
from torchvision.models.densenet import DenseNet121_Weights, densenet121
from torchvision.models.mobilenetv2 import mobilenet_v2,MobileNet_V2_Weights
from prior_box import PriorBox

class RetinaNet(nn.Module):

    def __init__(self, config, pretrained=True):
        super().__init__()

        # Feature Pyramid Network (FPN) with four feature maps of resolutions
        # 1/4, 1/8, 1/16, 1/32 and `num_filters` filters for all feature maps.
        num_anchors = config.get('num_anchors', 6)
        num_filters_fpn = config.get('num_filters_fpn', 128)
        self.num_classes = config['num_classes']
        fmaps = [56, 56, 28, 14, 7]
        self.size = config["img_size"]
        self.priorbox = PriorBox(self.size, feature_maps=fmaps)
        self.num_anchors = num_anchors
        self.fpn = FPN(out_channels=num_anchors * (4 + self.num_classes))

        with torch.no_grad():
            self.priors = self.priorbox.forward()
            if torch.cuda.is_available():
                self.priors = self.priors.cuda()

    def forward(self, x):
        maps = self.fpn(x)
        loc = list()
        conf = list()
        for map in maps:
            loc.append(map[:, :self.num_anchors * 4].permute(0, 2, 3, 1).contiguous())
            conf.append(map[:, self.num_anchors * 4:].permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        output = (
            loc.view(loc.size(0), -1, 4),
            conf.view(conf.size(0), -1, self.num_classes),
            self.priors
        )
        return output


class FPN(nn.Module):

    def __init__(self, out_channels):

        super().__init__()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        # backbone = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1).features
        # self.backbones = nn.ModuleList([
        #     backbone[:4],
        #     backbone.denseblock1,
        #     nn.Sequential(
        #         backbone.transition1,
        #         backbone.denseblock2,
        #     ),
        #     nn.Sequential(
        #         backbone.transition2,
        #         backbone.denseblock3,
        #     ),
        #     nn.Sequential(
        #         backbone.transition3,
        #         backbone.denseblock4,
        #     )
        # ])
        backbone_mobile = mobilenet_v2(MobileNet_V2_Weights).features
        self.backbones = nn.ModuleList([
            nn.Sequential(
                backbone_mobile[:3],
                nn.Conv2d(24, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU6(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(64, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                backbone_mobile[3:8],
                nn.ConvTranspose2d(64, 256, kernel_size=4, stride=4, padding=0, bias=False)
            ),
            nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0),
                backbone_mobile[8:11],
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 512, kernel_size=1, stride=1, padding=0),
            ),
            nn.Sequential(
                nn.Conv2d(512, 64, kernel_size=3, stride=2, padding=1),  # Reduce spatial dimensions to 14x14
                nn.ReLU(inplace=True),
                backbone_mobile[11:15],
                nn.ConvTranspose2d(160, 512, kernel_size=2, stride=2),  # Upsample to 14x14
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0),  # 1x1 convolution to change channel size
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(1024, 160, kernel_size=2, stride=2),  # Reduce spatial dimensions to 7x7
                nn.ReLU(inplace=True),
                backbone_mobile[15:],
                nn.Conv2d(1280, 1024, kernel_size=1, stride=1, padding=0)
                
            ) 
        ])

        self.up1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(128 + 64, 128, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.conv0 = nn.Sequential(
            nn.Conv2d(1024, out_channels, kernel_size=1),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, out_channels, kernel_size=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, out_channels, kernel_size=1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, out_channels, kernel_size=1),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, out_channels, kernel_size=1),
        )

    def forward(self, x):
        # Bottom-up pathway, from ResNet
        enc0 = self.backbones[0](x)
        enc1 = self.backbones[1](enc0)  # 256
        enc2 = self.backbones[2](enc1)  # 512
        enc3 = self.backbones[3](enc2)  # 1024
        enc4 = self.backbones[4](enc3)  # 2048

        up1 = self.upsample(enc4)
        up1 = up1 + enc3
        up1 = self.up1(up1)

        up2 = self.upsample(up1)
        up2 = up2 + enc2
        up2 = self.up2(up2)

        up3 = self.upsample(up2)
        up3 = up3 + enc1
        up3 = self.up3(up3)

        up4 = torch.cat([up3, enc0], 1)
        up4 = self.up4(up4)

        map1 = self.conv0(enc4)
        map2 = self.conv1(up1)
        map3 = self.conv2(up2)
        map4 = self.conv3(up3)
        map5 = self.conv4(up4)
        # for i in [map1, map2, map3, map4, map5]:
        #     print(i.size())
        return map1, map2, map3, map4, map5


def build_retinanet(config):
    return nn.DataParallel(RetinaNet(config))