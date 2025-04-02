# from dataset.brats_data_utils_multi_label import get_loader_brats
# from light_training.evaluation.metric import dice
# from light_training.trainer import Trainer
# from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
# from light_training.utils.files_helper import save_new_model_and_delete_last
import torch
import torch.nn as nn
from monai.utils import set_determinism
from monai.inferers import SlidingWindowInferer
import sys
# module_path = '/home/qitam/sdb2/home/qiteam_project/huang/SSL4MIS-master/code/networks'
# if module_path not in sys.path:
#     sys.path.append(module_path)
from unet_diff_util.basic_unet_denose import BasicUNetDe
from unet_diff_util.basic_unet import BasicUNetEncoder,BasicUNetEncoder_multi
import argparse
from monai.losses.dice import DiceLoss
import yaml
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler
set_determinism(123)
import os

number_modality = 4
number_targets = 3 ## WT, TC, ET
# number_targets = 1 ## WT, TC, ET
seg_num = 4
class DiffUNet(nn.Module):
    def __init__(self,args) -> None:
        super().__init__()
        # self.args = args
        self.embed_model = BasicUNetEncoder(3, number_modality, number_targets, [64, 64, 128, 256, 512, 64])
        # self.embed_model = BasicUNetEncoder_multi(3, number_modality, number_targets, [64, 64, 128, 256, 512, 64])

        self.model = BasicUNetDe(3, number_modality + number_targets, number_targets, [64, 64, 128, 256, 512, 64],
                                 act=("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))

        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [1000]),
                                         betas=betas,
                                         model_mean_type=ModelMeanType.START_X,
                                         model_var_type=ModelVarType.FIXED_LARGE,
                                         loss_type=LossType.MSE,
                                         )

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [50]),
                                                betas=betas,
                                                model_mean_type=ModelMeanType.START_X,
                                                model_var_type=ModelVarType.FIXED_LARGE,
                                                loss_type=LossType.MSE,
                                                )
        self.sampler = UniformSampler(1000)

    def forward(self, image=None, x=None, pred_type=None, step=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            embeddings = self.embed_model(image)
            return self.model(x, t=step, image=image, embeddings=embeddings)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)
            # batch = self.args.batch_size-self.args.labeled_bs
            # sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (batch, number_targets, 96, 96, 96),
            #                                                     model_kwargs={"image": image, "embeddings": embeddings})
            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, number_targets, 96, 96, 96),
                                                                model_kwargs={"image": image, "embeddings": embeddings})
            sample_out = sample_out["pred_xstart"]
            return sample_out
