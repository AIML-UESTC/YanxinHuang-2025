from networks.unet_3D import unet_3D
from networks.vnet import VNet
from networks.VoxResNet import VoxResNet
from networks.attention_unet import Attention_UNet
from networks.nnunet import initialize_network
from networks.unet_3D_multi import unet_3D_multi,unet_3D_multi_feature
from networks.unet_multi_out import unet_multi_out,multi_Encoder,shared_Decoder,shared_Decoder_con
from networks.diffusion import DiffUNet

def net_factory_3d(net_type="unet_3D", in_chns=1, class_num=2,args = None):
    if net_type == "unet_3D":
        net = unet_3D(n_classes=class_num, in_channels=in_chns).cuda()
    elif net_type == "attention_unet":
        net = Attention_UNet(n_classes=class_num, in_channels=in_chns).cuda()
    elif net_type == "voxresnet":
        net = VoxResNet(in_chns=in_chns, feature_chns=64,
                        class_num=class_num).cuda()
    elif net_type == "vnet":
        net = VNet(n_channels=in_chns, n_classes=class_num,
                   normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "nnUNet":
        net = initialize_network(num_classes=class_num).cuda()
    elif net_type == "unet_3D_multi":
        net = unet_3D_multi(n_classes=class_num, in_channels=in_chns).cuda()
    elif net_type == "unet_3D_multi_feature":
        net = unet_3D_multi_feature(n_classes=class_num, in_channels=in_chns).cuda()
    elif net_type == "unet_multi_out":
        # net = unet_multi_out(n_classes=class_num, in_channels=in_chns).cuda()
        net = (multi_Encoder(n_classes=class_num, in_channels=in_chns).cuda(),shared_Decoder(n_classes=class_num, in_channels=in_chns).cuda())
    elif net_type == "unet_feature_con":
        net = (multi_Encoder(n_classes=class_num, in_channels=in_chns).cuda(),shared_Decoder_con(n_classes=class_num, in_channels=4).cuda())
    elif net_type == "diff":
        net = DiffUNet(args)
    else:
        net = None
    return net
