import timm
import torch
import torch.nn as nn
import torchvision
from torchvision.models.segmentation import deeplabv3_resnet101


####### UNet ########
class U_block(nn.Module):
    '''building block of each stage'''

    def __init__(self, in_chan, out_chan) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 3),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(True),
            nn.Conv2d(out_chan, out_chan, 3),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class U_Decoder(nn.Module):
    '''Decoder of U-Net'''

    def __init__(self, chans) -> None:
        super().__init__()
        self.chans = chans
        self.blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for idx in range(len(chans) - 1):
            self.up_convs.append(nn.ConvTranspose2d(
                chans[idx], chans[idx + 1], 2, 2))
            self.blocks.append(U_block(2 * chans[idx + 1], chans[idx + 1]))

    def forward(self, skip_cons, x, skip_mask=None):
        def crop_feature(feature, shape):
            _, _, H, W = shape
            return torchvision.transforms.CenterCrop([H, W])(feature)

        if skip_mask is None:
            skip_mask = [True] * len(self.blocks)

        for i, (block, up_conv, feature) in enumerate(zip(self.blocks, self.up_convs, skip_cons)):
            x = up_conv(x)
            feature = crop_feature(feature, x.shape)
            if not skip_mask[i]:
                feature = torch.zeros_like(feature) # remove one skip connection 
            x = torch.cat((feature, x), dim=1)
            x = block(x)

        return x


class U_Net(nn.Module):
    def __init__(
        self,
        n_class=7,
        skip_idx: int = -1
    ) -> None:
        super().__init__()
        self.Encoder = timm.create_model(
            'resnet34', features_only=True, pretrained=True)
        dec_chans = self.Encoder.feature_info.channels()
        print("U_net chans:", dec_chans)
        dec_chans.reverse()
        self.center_block = nn.Sequential(
            nn.Conv2d(dec_chans[0], dec_chans[0], 3, padding=1),
            nn.BatchNorm2d(dec_chans[0]),
            nn.ReLU(True)
        )
        self.Decoder = U_Decoder(dec_chans)
        self.clf = nn.Conv2d(dec_chans[-1], n_class, 1)
        # --- build a fixed skip mask if skip_idx is provided ---
        self.skip_mask = None
        if skip_idx != -1:
            n = len(self.Decoder.blocks)             # number of decoder stages
            assert 0 <= skip_idx < n
            self.skip_mask = [i != skip_idx for i in range(n)]  # False at the one to drop

        print("U_net #param:", sum(p.numel() for p in self.parameters()))

    def forward(self, x):
        def crop_feature(feature, shape):
            _, _, H, W = shape
            return torchvision.transforms.CenterCrop([H, W])(feature)
        in_shape = x.shape[:]

        features = self.Encoder(x)
        features.reverse()  # from down to top
        skips = features[1:]
        x = features[0]
        x = self.center_block(x)
        # bottom feature map doesn't need concatenation
        x = self.Decoder(skips, x, self.skip_mask)
        x = self.clf(x)
        x = crop_feature(x, in_shape)
        return x


######## DeepLabv3 resnet101 #########
class SegmentationModel(nn.Module):
    def __init__(self, num_classes):
        super(SegmentationModel, self).__init__()
        # use pre-trained ResNet50 with DeepLabV3
        self.model = deeplabv3_resnet101(weights=True)
        # Reduce the dimension from 256 to 7(num_classes)
        self.model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.model(x)['out']


if __name__ == '__main__':
    model_unet = U_Net(n_class=7, skip_idx=-1)
    print(f"the model architecture of unet: {model_unet}")

    model_deeplabv3 = SegmentationModel(num_classes=7)
    print(f"the model architecture of deeplabv3: {model_deeplabv3}")