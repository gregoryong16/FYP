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
        self.fpn = FPN(out_channels=num_anchors * (4 + self.num_classes),backbone='mobilenet_v2_deep')

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

    def __init__(self, out_channels, backbone="densenet"):

        super().__init__()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.backbone = backbone
        if backbone == "densenet":
            backbone = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1).features
            self.backbones = nn.ModuleList([
                backbone[:4],
                backbone.denseblock1,
                nn.Sequential(
                    backbone.transition1,
                    backbone.denseblock2,
                ),
                nn.Sequential(
                    backbone.transition2,
                    backbone.denseblock3,
                ),
                nn.Sequential(
                    backbone.transition3,
                    backbone.denseblock4,
                )
            ])
            self.enc0_channel = 64
            self.enc1_channel = 256
            self.enc2_channel = 512
            self.enc3_channel = 1024
            self.enc4_channel = 1024

        elif backbone == "mobilenet_v2":
            backbone_mobile = mobilenet_v2(MobileNet_V2_Weights).features
            self.backbones = nn.ModuleList([
                nn.Sequential(
                    backbone_mobile[:3], # out channels: 24
                ),
                nn.Sequential(
                    backbone_mobile[3:4], # out channels: 24
                ),
                nn.Sequential(
                    backbone_mobile[4:7], # out channels: 32
                ),
                nn.Sequential(
                    backbone_mobile[7:14], # out channels: 96
                ),
                nn.Sequential(
                    backbone_mobile[14:], # out channels: 1280
                )
            ])

            self.transform_enc4_to_enc3 = nn.Sequential(
                torch.nn.Conv2d(1280, 96, kernel_size=1, stride=1, padding=0),
                torch.nn.ReLU6(inplace=True)
            )

            self.enc0_channel = 24
            self.enc1_channel = 24
            self.enc2_channel = 32
            self.enc3_channel = 96
            self.enc4_channel = 1280
        elif backbone == "mobilenet_v2_deep":
            backbone_mobile = mobilenet_v2(MobileNet_V2_Weights).features
            self.backbones = nn.ModuleList([
                nn.Sequential(
                backbone_mobile[:3],
                nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels=24, out_channels=3, kernel_size=1),
                backbone_mobile[:3],
                ),
                nn.Sequential(
                    backbone_mobile[3:4],
                    backbone_mobile[3:4],
                ),
                nn.Sequential(
                    backbone_mobile[4:7],
                    nn.Upsample(size=(56, 56), mode='bilinear', align_corners=False),
                    nn.Conv2d(in_channels=32, out_channels=24, kernel_size=1),
                    backbone_mobile[4:7],
                ),
                nn.Sequential(
                    backbone_mobile[7:14],
                    nn.Upsample(size=(28,28), mode='bilinear', align_corners=False),
                    nn.Conv2d(in_channels=96, out_channels=32, kernel_size=1),
                    backbone_mobile[7:14],
                ),
                nn.Sequential(
                    backbone_mobile[14:],
                    nn.Upsample(size=(14,14), mode='bilinear', align_corners=False),
                    nn.Conv2d(in_channels=1280, out_channels=96, kernel_size=1),
                    backbone_mobile[14:],
                )
            ])

            self.transform_enc4_to_enc3 = nn.Sequential(
                torch.nn.Conv2d(1280, 96, kernel_size=1, stride=1, padding=0),
                torch.nn.ReLU6(inplace=True)
            )

            self.enc0_channel = 24
            self.enc1_channel = 24
            self.enc2_channel = 32
            self.enc3_channel = 96
            self.enc4_channel = 1280

        else:
            raise f"{backbone} not implemented."

        self.up1 = nn.Sequential(
            nn.Conv2d(self.enc3_channel, self.enc2_channel, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(self.enc2_channel),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(self.enc2_channel, self.enc1_channel, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(self.enc1_channel),
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(self.enc1_channel, self.enc0_channel, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(self.enc0_channel),
            nn.ReLU(inplace=True),
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(self.enc0_channel + self.enc0_channel, self.enc0_channel, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(self.enc0_channel),
            nn.ReLU(inplace=True),
        )

        self.conv0 = nn.Sequential(
            nn.Conv2d(self.enc4_channel, out_channels, kernel_size=1),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.enc2_channel, out_channels, kernel_size=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.enc1_channel, out_channels, kernel_size=1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.enc0_channel, out_channels, kernel_size=1),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.enc0_channel, out_channels, kernel_size=1),
        )

        

    def forward(self, x):
        # Bottom-up pathway, from ResNet
        enc0 = self.backbones[0](x)     # bs, channel_enc0, 56, 56
        enc1 = self.backbones[1](enc0)  # bs, channel_enc1, 56, 56
        enc2 = self.backbones[2](enc1)  # bs, channel_enc2, 28, 28
        enc3 = self.backbones[3](enc2)  # bs, channel_enc3, 14, 14
        enc4 = self.backbones[4](enc3)  # bs, channel_enc4, 7, 7

        up1 = self.upsample(enc4)  # bs, channel_enc4, 14, 14
        if up1.size(1) != enc3.size(1):
            # transform up1's channel size when channel_enc3 != channel_enc4
            up1 = self.transform_enc4_to_enc3(up1) + enc3
        else:
            up1 = up1 + enc3
        up1 = self.up1(up1)  # bs, channel_enc2, 14, 14

        up2 = self.upsample(up1)  # bs, channel_enc2, 28, 28
        up2 = up2 + enc2
        up2 = self.up2(up2)  # bs, channel_enc1, 28, 28

        up3 = self.upsample(up2)  # bs, channel_enc1, 56, 56
        up3 = up3 + enc1
        up3 = self.up3(up3)  # bs, channel_enc1, 56, 56

        up4 = torch.cat([up3, enc0], 1)  # bs, channel_enc1 + channel_enc0, 56, 56
        up4 = self.up4(up4)

        map1 = self.conv0(enc4) # torch.Size([1, 36, 7, 7])
        map2 = self.conv1(up1) # torch.Size([1, 36, 14, 14])
        map3 = self.conv2(up2) # torch.Size([1, 36, 28, 28])
        map4 = self.conv3(up3) # torch.Size([1, 36, 56, 56])
        map5 = self.conv4(up4) # torch.Size([1, 36, 56, 56])
        # for i in [map1, map2, map3, map4, map5]:
        #     print(i.size())
        return map1, map2, map3, map4, map5


def build_retinanet(config):
    return nn.DataParallel(RetinaNet(config))

if __name__=="__main__":
    model = FPN(out_channels=6 * (4 + 2), backbone="mobilenet_v2_deep")
    test_image = torch.rand(1,3,224,224)
    output_maps = model(test_image)
    for each_map in output_maps:
        print(each_map.size())
