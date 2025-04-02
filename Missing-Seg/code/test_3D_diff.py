import argparse
import os
import shutil
from glob import glob
from monai import transforms as transforms
from monai import transforms as mon_transforms
import torch

from networks.unet_3D import unet_3D
from networks.unet_3D_multi import unet_3D_multi
# from test_3D_util import test_all_case
from test_3D_diff_util import test_all_case
import os
from networks.diffusion import DiffUNet
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/root/autodl-tmp/BraTS2023-GLITraining', help='Name of Experiment')
# parser.add_argument('--root_path', type=str,
#                     default='/home/qitam/sdb2/home/qiteam_project/huang/SSL4MIS-master/data/BraTSmulti/BraTS2023-GLITraining', help='Name of Experiment')

parser.add_argument('--exp', type=str,
                    default='2023_diff', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='diff', help='model_name')
parser.add_argument('--mod', type=str,
                    default='multi_concat', help='experiment_mod')


def load_state_dict(model, weight_path, strict=True):
    sd = torch.load(weight_path, map_location="cpu")
    if "module" in sd:
        sd = sd["module"]
    new_sd = {}
    for k, v in sd.items():
        k = str(k)
        new_k = k[7:] if k.startswith("module") else k
        new_sd[new_k] = v

    model.load_state_dict(new_sd, strict=strict)

    print(f"model parameters are loaded successed.")
def Inference(FLAGS):
    transform = transforms.Compose([
        # mon_transforms.ConvertToMultiChannelBasedOnBratsClassesD(keys=["label"]),
        mon_transforms.CropForegroundd(keys=["image", "label"], source_key="image"),

        mon_transforms.RandSpatialCropd(keys=["image", "label"], roi_size=[96, 96, 96],
                                        random_size=False),
        mon_transforms.SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),
        mon_transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        mon_transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        mon_transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        mon_transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),

        mon_transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        mon_transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        mon_transforms.ToTensord(keys=["image", "label"], ),
    ])
    if FLAGS.mod == 'multi_concat':
        snapshot_path = "../model/{}/{}".format(FLAGS.exp, FLAGS.model)
        num_classes = 3
        test_save_path = "../model/{}/4class_ET_30000iter".format(FLAGS.exp)
        if os.path.exists(test_save_path):
            shutil.rmtree(test_save_path)
        os.makedirs(test_save_path)

        net = DiffUNet(parser)
        # save_mode_path = os.path.join(
        #     snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
        save_mode_path = "/root/SSL4MIS-master/model/2023_diff/diff/iter_30000.pth"
        load_state_dict(net,save_mode_path)
        print("init weight from {}".format(save_mode_path))
        net.eval()
        avg_metric = test_all_case(net, base_dir=FLAGS.root_path, method=FLAGS.model, test_list="test.txt", num_classes=num_classes,
                                   patch_size=(96, 96, 96), stride_xy=64, stride_z=64, test_save_path=test_save_path,mod=FLAGS.mod,transform=transform)

    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)
