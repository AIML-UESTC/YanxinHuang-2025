import argparse
import os
import shutil
from glob import glob

import torch

from networks.unet_3D import unet_3D
from networks.unet_3D_multi import unet_3D_multi
from test_3D_util import test_all_case_multi
from networks.unet_multi_out import unet_multi_out,multi_Encoder,shared_Decoder,shared_Decoder_con
import os
import copy
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/qitam/sdb2/home/qiteam_project/huang/SSL4MIS-master/data/BraTSmulti/BraTS2023-GLITraining', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='feature_consistency_concat', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_feature_con', help='model_name')
parser.add_argument('--mod', type=str,
                    default='multi_concat', help='experiment_mod')

def Inference(FLAGS):
    mod = FLAGS.mod
    if FLAGS.mod == 'multi_concat':
        snapshot_path = "../model/{}/{}".format(FLAGS.exp, FLAGS.model)
        num_classes = 2
        test_save_path = "../model/{}/Prediction_multi_out".format(FLAGS.exp)
        if os.path.exists(test_save_path):
            shutil.rmtree(test_save_path)
        os.makedirs(test_save_path)
        # net = unet_3D(n_classes=num_classes, in_channels=1).cuda()
        # net = unet_3D_multi(n_classes=num_classes, in_channels=1).cuda()
        net = (multi_Encoder(n_classes=num_classes, in_channels=1).cuda(),
         shared_Decoder_con(n_classes=num_classes, in_channels=4).cuda())
        encoders = [copy.deepcopy(net[0]) for _ in range(4)]
        decoder = net[1]
        save_encoder_1_path = os.path.join(
            snapshot_path, 'E_0_{}_best_model.pth'.format(FLAGS.model))
        save_encoder_2_path = os.path.join(
            snapshot_path, 'E_1_{}_best_model.pth'.format(FLAGS.model))
        save_encoder_3_path = os.path.join(
            snapshot_path, 'E_2_{}_best_model.pth'.format(FLAGS.model))
        save_encoder_4_path = os.path.join(
            snapshot_path, 'E_3_{}_best_model.pth'.format(FLAGS.model))
        save_decoder_path = os.path.join(
            snapshot_path, 'D_{}_best_model.pth'.format(FLAGS.model))
        # save_mode_path = os.path.join(
        #     snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
        encoders[0].load_state_dict(torch.load(save_encoder_1_path))
        encoders[1].load_state_dict(torch.load(save_encoder_2_path))
        encoders[2].load_state_dict(torch.load(save_encoder_3_path))
        encoders[3].load_state_dict(torch.load(save_encoder_4_path))
        decoder.load_state_dict(torch.load(save_decoder_path))
        print("init weight from {}".format(save_encoder_1_path))
        # net.eval()
        [encoder.eval() for encoder in encoders]
        decoder.eval()
        multi_model = [encoders, decoder]
        avg_metric = test_all_case_multi(multi_model, base_dir=FLAGS.root_path, method=FLAGS.model, test_list="test.txt", num_classes=num_classes,
                                   patch_size=(96, 96, 96), stride_xy=64, stride_z=64, test_save_path=test_save_path,mod=FLAGS.mod)

    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)
