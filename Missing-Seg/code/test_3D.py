import argparse
import os
import shutil
from glob import glob

import torch

from networks.unet_3D import unet_3D
from networks.unet_3D_multi import unet_3D_multi
# from test_3D_util import test_all_case
from test_3D_multi_class_util import test_all_case
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/qitam/sdb2/home/qiteam_project/huang/SSL4MIS-master/data/BraTSmulti/BraTS2023-GLITraining', help='Name of Experiment')
# parser.add_argument('--root_path', type=str,
#                     default='/root/autodl-tmp/BraTS2023-GLITraining', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='2023_cp_supply_core_region', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_3D_multi', help='model_name')
parser.add_argument('--mod', type=str,
                    default='multi_concat', help='experiment_mod')

def Inference(FLAGS):
    mod = FLAGS.mod
    if 'multi' not in FLAGS.mod:
        snapshot_path = "../model/{}/{}".format(FLAGS.exp, FLAGS.model)
        num_classes = 2
        test_save_path = "../model/{}/Prediction".format(FLAGS.exp)
        if os.path.exists(test_save_path):
            shutil.rmtree(test_save_path)
        os.makedirs(test_save_path)
        net = unet_3D(n_classes=num_classes, in_channels=1).cuda()
        save_mode_path = os.path.join(
            snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
        net.load_state_dict(torch.load(save_mode_path))
        print("init weight from {}".format(save_mode_path))
        net.eval()
        avg_metric = test_all_case(net, base_dir=FLAGS.root_path, method=FLAGS.model, test_list="test.txt", num_classes=num_classes,
                                   patch_size=(96, 96, 96), stride_xy=64, stride_z=64, test_save_path=test_save_path,mod=FLAGS.mod)
    elif FLAGS.mod == 'multi_concat':
        snapshot_path = "../model/{}/{}".format(FLAGS.exp, FLAGS.model)
        num_classes = 4
        test_save_path = "../model/{}/cp_supply".format(FLAGS.exp)
        print('任务是:',test_save_path.split('/')[-1])
        if os.path.exists(test_save_path):
            shutil.rmtree(test_save_path)
        os.makedirs(test_save_path)
        # net = unet_3D(n_classes=num_classes, in_channels=1).cuda()
        net = unet_3D_multi(n_classes=num_classes, in_channels=1).cuda()
        save_mode_path = os.path.join(
            snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
        # save_mode_path = '/home/qitam/sdb2/home/qiteam_project/huang/SSL4MIS-master/model/semi_10/iter_12000.pth'
        # save_mode_path = '/root/SSL4MIS-master/model/2023_semi_10_1/unet_3D_multi/iter_26000.pth'
        # save_mode_path = '/home/qitam/sdb2/home/qiteam_project/huang/SSL4MIS-master/model/2023_semi_50/unet_3D_multi/iter_17400_dice_0.6468.pth'
        net.load_state_dict(torch.load(save_mode_path))
        print("init weight from {}".format(save_mode_path))
        net.eval()
        avg_metric = test_all_case(net, base_dir=FLAGS.root_path, method=FLAGS.model, test_list="test.txt", num_classes=num_classes,
                                   patch_size=(96, 96, 96), stride_xy=64, stride_z=64, test_save_path=test_save_path,mod=FLAGS.mod)

    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)
