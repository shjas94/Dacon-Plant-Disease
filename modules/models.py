import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class EffnetV2(nn.Module):
    def __init__(self, weight_name, num_classes):
        super(EffnetV2, self).__init__()
        self.effnet = timm.create_model(
            weight_name, pretrained=True, num_classes=num_classes)
        # tf_efficientnetv2_m_in21ft1k

    def forward(self, x):
        out = self.effnet(x)
        return out


class Effnet(nn.Module):
    def __init__(self, weight_name, num_classes):
        super(Effnet, self).__init__()
        self.effnet = timm.create_model(
            weight_name, pretrained=True, num_classes=num_classes)
        # tf_efficientnet_b7_ns

    def forward(self, x):
        out = self.effnet(x)
        return out


class Swin(nn.Module):
    def __init__(self, weight_name, num_classes):
        super(Swin, self).__init__()
        self.swin = timm.create_model(
            weight_name, pretrained=True, num_classes=num_classes)
    # swin_large_patch4_window12_384

    def forward(self, x):
        out = self.swin(x)
        return out


class Encoder(nn.Module):
    def __init__(self, encoder_name, weight_name, num_classes):
        super(Encoder, self).__init__()
        if encoder_name == "EfficientNet":
            self.model = Effnet(weight_name=weight_name,
                                num_classes=num_classes)
        elif encoder_name == "EfficientNetV2":
            self.model = EffnetV2(weight_name=weight_name,
                                  num_classes=num_classes)
        elif encoder_name == "Swin":
            self.model = Swin(weight_name=weight_name, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
