import numpy as np
import torch
import math
import SimpleITK as sitk
from tqdm import tqdm
import nibabel as nib
from medpy import metric
from monai.inferers import SlidingWindowInferer
import os
from monai import transforms
from medpy import metric
from light_training.evaluation.metric import dice, hausdorff_distance_95, recall, fscore
def cal(o,t):
    tc_ravd = 0.0
    tc_hd = 0.0
    tc_asd = 0.0
    #o是预测值，t是真实值
    if np.sum(o) != 0 and np.sum(t) != 0:
        tc = dice(o, t)
        # tc_ravd = abs(metric.ravd(o, t))
        # tc_hd = metric.hd95(o, t)
        # tc_asd = metric.asd(o, t)
    elif np.sum(o) == 0 and np.sum(t) == 0:
        tc = 1.0
        tc_ravd = 0.0
        tc_hd = 0.0
        tc_asd = 0.0
    else :
        tc = None
        tc_ravd = None
        tc_hd = None
        tc_asd = None

    return np.array([tc, tc_ravd, tc_hd, tc_asd])
def process_image(sample, net, patch_size, stride_xy, stride_z, num_classes, test_save_path, ids):
    # 假设 ET、TC、WT 的类别索引

    class_et = 1  # 增强型肿瘤
    class_tc = 2  # 肿瘤核心
    class_wt = 3  # 整体肿瘤
    image = sample['image']
    label = sample['label']
    c, w, h, d = image.shape

    # 如果需要填充
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant',
                       constant_values=0)

    cc, ww, hh, dd = image.shape

    # 计算每个维度的补丁数量
    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

    # 初始化分数图和计数图
    score_map = np.zeros((num_classes,) + (ww, hh, dd)).astype(np.float32)
    cnt = np.zeros((ww, hh, dd)).astype(np.float32)

    # output = window_infer(image.unsqueeze(0), net, pred_type="ddim_sample")
    # return output
    # 滑动窗口推理
    for x in range(sx):
        print('x',x)
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(sy):
            print('y', y)
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(sz):
                print('z', z)
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch = np.expand_dims(test_patch, axis=0).astype(np.float32)
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                test_patch = torch.from_numpy(test_patch).to(device)

                with torch.no_grad():
                    y1 = net(test_patch,pred_type="ddim_sample")
                    y_1 = torch.softmax(y1, dim=1)
                y_1 = y_1.cpu().data.numpy()
                y_1 = y_1[0, :, :, :, :]
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] += y_1
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] += 1
    print('推理结束')
    # 避免除零错误
    cnt[cnt == 0] = 1
    # 归一化分数图
    score_map /= np.expand_dims(cnt, axis=0)

    # 提取各部分
    label_map = np.argmax(score_map, axis=0)

    # 提取每个部分的分数图
    et_map = score_map[class_et-1, :, :, :]
    tc_map = score_map[class_tc-1, :, :, :]
    wt_map = score_map[class_wt-1, :, :, :]

    # 如果需要，去除填充
    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        et_map = et_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        tc_map = tc_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        wt_map = wt_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]


    return label_map, et_map, tc_map, wt_map


def test_all_case(net, base_dir, method="unet_3D", test_list="full_test.list", num_classes=4, patch_size=(48, 160, 160),
                  stride_xy=32, stride_z=24, test_save_path=None, transform = None,mod=None):
    with open(base_dir + '/{}'.format(test_list), 'r') as f:
        image_list = f.readlines()

    window_infer = SlidingWindowInferer(roi_size=[96, 96, 96],
                                        sw_batch_size=1,
                                        overlap=0.5)
    image_list = [base_dir + "/{}".format(item.replace('\n', '').split(",")[0]) for item in image_list]
    total_metric = np.zeros((3, 4))  # 3 个部分：ET、TC、WT
    print("测试开始")

    # 用于保存指标的文件
    with open(test_save_path + "/{}.txt".format(method), "a") as f:
        for image_path in tqdm(image_list):
            ids = image_path.split("/")[-1].replace(".h5", "")
            label_path = image_path.replace('t1n', 'seg')
            try:
                T1 = nib.load(image_path).get_fdata()
                T1ce = nib.load(image_path.replace(image_path[:-7].split('-')[-1], 't1c')).get_fdata()
                T2w = nib.load(image_path.replace(image_path[:-7].split('-')[-1], 't2w')).get_fdata()
                T2f = nib.load(image_path.replace(image_path[:-7].split('-')[-1], 't2f')).get_fdata()
                """
                测试分为三种情况，1.全模态。2.缺失模态，这里选择零填充处理缺失模态的情形。3.伪模态补全，这里用加权补全。
                """
                # T2f = np.zeros_like(T1)
                # T2f = (T1 + T1ce + T2w) / 3
                image_data = np.array([T1, T1ce, T2w, T2f]).astype(np.float32)

                label = nib.load(label_path).get_fdata()
                # seg_data = np.expand_dims(np.array(label).astype(np.int32), axis=0)
            except Exception as e:
                print(f"加载图像时出错: {e}")
                continue
            label = [(label == 1) | (label == 3), (label == 1) | (label == 3) | (label == 2),label == 3]
            label = np.stack(label, axis=0).astype(np.float32)
            val = {"image": image_data, "label": label}
            val_transform = transforms.Compose([
                # transforms.ConvertToMultiChannelBasedOnBratsClassesD(keys=["label"]),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.ToTensord(keys=["image", "label"]),
            ]
            )
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

            val = val_transform(val)
            # val = {
            #     x: val[x].to(device)
            #     for x in val if isinstance(val[x], torch.Tensor)
            # }
            net.to(device)
            trans_input = val["image"]
            trans_label = val["label"]
            net_input = torch.tensor(np.expand_dims(trans_input,axis=0)).cuda()
            # output = net(val['image'].unsqueeze(0), pred_type="ddim_sample")
            # output,et_map, tc_map, wt_map = process_image(val, net, patch_size, stride_xy, stride_z, num_classes,
            #                                                   test_save_path, ids)
            output = window_infer(net_input, net, pred_type="ddim_sample")
            output = torch.sigmoid(output)
            print('推理完成')
            # output = (output > 0.5).float().cpu().numpy()
            # target = label.cpu().numpy()
            output = (output > 0.5).float().cpu().numpy()
            target = np.expand_dims(trans_label,axis=0)

            o = output[:, 1]
            t = target[:, 1]  # ce
            sitk.WriteImage(sitk.GetImageFromArray(o.squeeze().astype(np.uint8)),
                            f"{test_save_path}/{ids}_wt_pre.nii.gz")
            sitk.WriteImage(sitk.GetImageFromArray(t.squeeze().astype(np.uint8)),
                            f"{test_save_path}/{ids}_wt_gt.nii.gz")
            wt = cal(o,t)

            # core
            o = output[:, 0]
            t = target[:, 0]
            tc = cal(o,t)
            sitk.WriteImage(sitk.GetImageFromArray(o.squeeze().astype(np.uint8)),
                            f"{test_save_path}/{ids}_tc_pre.nii.gz")
            sitk.WriteImage(sitk.GetImageFromArray(t.squeeze().astype(np.uint8)),
                            f"{test_save_path}/{ids}_tc_gt.nii.gz")

            # active
            o = output[:, 2]
            t = target[:, 2]
            sitk.WriteImage(sitk.GetImageFromArray(o.squeeze().astype(np.uint8)),
                            f"{test_save_path}/{ids}_et_pre.nii.gz")
            sitk.WriteImage(sitk.GetImageFromArray(t.squeeze().astype(np.uint8)),
                            f"{test_save_path}/{ids}_et_gt.nii.gz")
            et = cal(o,t)

            metrics = np.array([et,tc,wt])
            print(f"wt is {wt}, tc is {tc}, et is {et}")
            # 可选：保存原始图像和标签图
            sitk.WriteImage(sitk.GetImageFromArray(T1), f"{test_save_path}/{ids}_img.nii.gz")
            # sitk.WriteImage(sitk.GetImageFromArray(output.astype(np.uint8)), f"{test_save_path}/{ids}_pre.nii.gz")
            # sitk.WriteImage(sitk.GetImageFromArray(label.astype(np.uint8)), f"{test_save_path}/{ids}_label.nii.gz")
            # 计算并保存指标
            if None not in metrics:
                total_metric += metrics
                for i in range(metrics.shape[0]):  # metrics.shape[0] 为 3
                    # 如果是第一行，写入 ids
                    if i == 0:
                        line = "{} {} {} {} {} \n".format(
                            ids,
                            metrics[i, 0], metrics[i, 1], metrics[i, 2], metrics[i, 3]
                        )
                    else:
                        # 否则用空格代替 ids
                        line = "       {} {} {} {} \n".format(
                            metrics[i, 0], metrics[i, 1], metrics[i, 2], metrics[i, 3]
                        )
                    # 将这一行写入文件
                    f.write(line)
            else:
                image_list.remove(image_path)
        # 计算并保存平均指标
        num_images = len(image_list)
        mean_metrics = total_metric / num_images
        f.writelines("Mean metrics,{},{},{},{}\n,{},{},{},{}\n,{},{},{},{}".format(
            mean_metrics[0, 0], mean_metrics[0, 1], mean_metrics[0, 2], mean_metrics[0, 3],
            mean_metrics[1, 0], mean_metrics[1, 1], mean_metrics[1, 2], mean_metrics[1, 3],
            mean_metrics[2, 0], mean_metrics[2, 1], mean_metrics[2, 2], mean_metrics[2, 3]
        ))
    print("测试完成，结果已保存")

    return total_metric/len(image_list)

# def calculate_metric_percase(label, label_map):
#     # label = label.unsqueeze(0)
#     def compute_metrics(pred, gt):
#         if np.sum(gt) != 0 and np.sum(pred)!=0:
#             dice = metric.dc(pred, gt)
#             ravd = abs(metric.ravd(pred, gt))
#             hd = metric.hd95(pred, gt)
#             asd = metric.asd(pred, gt)
#         else:
#             dice = 0.0
#             ravd = 0.0
#             hd = 0.0
#             asd = 0.0
#
#         return np.array([dice, ravd, hd, asd])
#
#     # Ground truth binary masks
#     # et_true = (label == 3).astype(np.float32)
#     # tc_true = (np.isin(label, [1, 3])).astype(np.float32)
#     # wt_true = (np.isin(label, [1, 2, 3])).astype(np.float32)
#     #
#     # et_true = label[:,0,:,:,:]
#     # tc_true = label[:,1,:,:,:]
#     # wt_true = label[:,2,:,:,:]
#     # et_pred = label_map[:,0,:,:,:]
#     # tc_pred = label_map[:,1,:,:,:]
#     # wt_pred = label_map[:,2,:,:,:]
#     #
#     # valid_labels = np.unique(label)
#     #
#     # # Create masks for valid labels in prediction
#     # et_pred = (label_map == 3).astype(np.float32) if 3 in valid_labels else np.zeros_like(label_map, dtype=np.float32)
#     # tc_pred = (np.isin(label_map, [1, 3])).astype(np.float32) if any(
#     #     l in valid_labels for l in [1, 3]) else np.zeros_like(label_map, dtype=np.float32)
#     # wt_pred = (np.isin(label_map, [1, 2, 3])).astype(np.float32) if any(
#     #     l in valid_labels for l in [1, 2, 3]) else np.zeros_like(label_map, dtype=np.float32)
#     #
#     # Calculate metrics for each class
#     et_metrics = compute_metrics(et_pred, et_true)
#     tc_metrics = compute_metrics(tc_pred, tc_true)
#     wt_metrics = compute_metrics(wt_pred, wt_true)
#
#
#     return np.array([et_metrics, tc_metrics, wt_metrics])

