import yaml
import torch.nn as nn
import torch.nn.functional as F

from hrnet.hrnet_ocr import get_seg_model
import segmentation_models_pytorch as smp


def init_models(model, encoder):
    if encoder == 'r152':
        encoder = 'resnet152'
    elif encoder == 'r101':
        encoder = 'resnet101'
    elif encoder == 'r50':
        encoder = 'resnet50'
    elif encoder == 'effb3':
        encoder = 'efficientnet-b3'
    elif encoder == 'effb5':
        encoder = 'efficientnet-b5'
    
    else:
        encoder = encoder
    
    # preprocess_input = get_preprocessing_fn(encoder, pretrained='imagenet')

    print(encoder)

    if model == "deeplabv3plus":
        model = smp.DeepLabV3Plus(
                    encoder_name = encoder,
                    encoder_weights = 'imagenet',
                    in_channels = 3,
                    classes = 29
                )
    elif model == 'hrnetocr':
        model = HRNetOCR()
    elif model == "deeplabv3":
        model = smp.DeepLabV3(
                    encoder_name = encoder,
                    encoder_weights = 'imagenet',
                    in_channels = 3,
                    classes = 29
                )
    elif model == "unet2plus":
        model = smp.UnetPlusPlus(
                    encoder_name = encoder,
                    encoder_weights = 'imagenet',
                    in_channels = 3,
                    classes = 29
                )
    elif model == "unet":
        model = smp.Unet(
                    encoder_name = encoder,
                    encoder_weights = 'imagenet',
                    in_channels = 3,
                    classes = 29
                )
    elif model == "pspnet":
        model = smp.PSPNet(
                    encoder_name = encoder,
                    encoder_weights = 'imagenet',
                    in_channels = 3,
                    classes = 29
                )
    return model




class HRNetOCR(nn.Module):
    def __init__(self, num_classes = 29, target_size = 1024):

        super().__init__()
        self.num_classes = num_classes
        self.w, self.h = target_size, target_size

        with open("./hrnet/hrnet_config/hrnet_ocr_w48.yaml", "r") as f:
            cfg = yaml.safe_load(f)

        self.model = get_seg_model(cfg)
    
    def forward(self, x):
        
        if self.training:
            x = self.model(x)
            x = [F.interpolate(input = x_, size = (self.w, self.h), mode = "bilinear", align_corners = True) for x_ in x]
            return x

        else:
            x = self.model(x)
            x = F.interpolate(input = x[0], size = (self.w, self.h), mode = "bilinear", align_corners = True)
        return x

