import numpy as np
import torch
import math
import SimpleITK as sitk
from tqdm import tqdm
import nibabel as nib
from medpy import metric


def get_cp_to_supply(volume_batch, label_batch):
    # 创建与 volume_batch 相同形状的输出张量
    output = volume_batch.clone()

    # 提取 T1, T1ce 和 T2w（要填充的）通道
    T1 = volume_batch[:, 0, :, :, :]
    T1ce = volume_batch[:, 1, :, :, :]
    # T2w_filled = volume_batch[:, 3, :, :, :]
    T2w_filled = torch.zeros_like(T1)
    # 定义立方体范围 [24:72, 24:72, 24:72]（用 T1 填充的区域）
    x_start, x_end = 16, 80
    y_start, y_end = 16, 80
    z_start, z_end = 16, 80

    # 先用 T1ce 填充整个 T2w_filled
    T2w_filled[:, :, :, :] = T1ce[:, :, :, :]

    # 获取病灶区域掩码（如果需要根据 label_batch 中非零区域替换）
    # lesion_mask = (label_batch[:, x_start:x_end, y_start:y_end, z_start:z_end] > 0)

    # 将病灶区域用 T1 填充
    T2w_filled[:, x_start:x_end, y_start:y_end, z_start:z_end] = T1[:, x_start:x_end, y_start:y_end, z_start:z_end]

    # 将非病灶区域用 T1 填充
    # T2w_filled[:, x_start:x_end, y_start:y_end, z_start:z_end] = T1ce[:, x_start:x_end, y_start:y_end, z_start:z_end]

    # 更新输出张量的第 3 通道
    output[:, 3, :, :, :] = T2w_filled

    return output
def process_image(image, net, patch_size, stride_xy, stride_z, num_classes, test_save_path, ids):
    # 假设 ET、TC、WT 的类别索引
    class_et = 1  # 增强型肿瘤
    class_tc = 2  # 肿瘤核心
    class_wt = 3  # 整体肿瘤

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

    # 滑动窗口推理
    for x in range(sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch = np.expand_dims(test_patch, axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1 = net(test_patch)
                    y = torch.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] += y
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] += 1

    # 避免除零错误
    cnt[cnt == 0] = 1
    # 归一化分数图
    score_map /= np.expand_dims(cnt, axis=0)

    # 提取各部分
    label_map = np.argmax(score_map, axis=0)

    # 提取每个部分的分数图
    et_map = score_map[class_et, :, :, :]
    tc_map = score_map[class_tc, :, :, :]
    wt_map = score_map[class_wt, :, :, :]

    # 如果需要，去除填充
    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        et_map = et_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        tc_map = tc_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        wt_map = wt_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]


    return label_map, et_map, tc_map, wt_map


def test_all_case(net, base_dir, method="unet_3D", test_list="full_test.list", num_classes=4, patch_size=(48, 160, 160),
                  stride_xy=32, stride_z=24, test_save_path=None, mod=None):
    with open(base_dir + '/{}'.format(test_list), 'r') as f:
        image_list = f.readlines()

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
                # T2f = nib.load(image_path.replace(image_path[:-7].split('-')[-1], 't2f')).get_fdata()
                # T2f = np.zeros_like(T1)


                # T2f = (T1+T1ce+T2w)/3

                T2f = T1ce.copy()
                # 将病灶区域用 T1 填充
                T2f[88:152, 88:152, 46:110] = T1[88:152, 88:152, 46:110]
            except Exception as e:
                print(f"加载图像时出错: {e}")
                continue
            #
            # image = np.stack((T1, T1ce, T2w, T2f), axis=0)
            try:
                label = nib.load(label_path).get_fdata()
            except Exception as e:
                print(f"加载标签时出错: {e}")
                continue

            # 确定病灶区域：根据 label_batch 中非零区域确定
            # lesion_mask = (label > 0)
            # # 将 T2w_filled 的病灶区域用 T1ce 填充，其他区域用 T1 填充
            # T2f[lesion_mask] = T1ce[lesion_mask]
            # T2f[~lesion_mask] = T1[~lesion_mask]

            image = np.stack((T1, T1ce, T2w, T2f), axis=0)
            # 调用图像处理函数
            pred, et_map, tc_map, wt_map = process_image(image, net, patch_size, stride_xy, stride_z, num_classes,
                                                              test_save_path, ids)

            # 保存图像

            # 可选：保存原始图像和标签图
            sitk.WriteImage(sitk.GetImageFromArray(T1ce), f"{test_save_path}/{ids}_img.nii.gz")
            sitk.WriteImage(sitk.GetImageFromArray(pred.astype(np.uint8)), f"{test_save_path}/{ids}_pre.nii.gz")
            sitk.WriteImage(sitk.GetImageFromArray(label.astype(np.uint8)), f"{test_save_path}/{ids}_label.nii.gz")

            # 计算并保存指标
            metrics = calculate_metric_percase(label, pred)
            total_metric += metrics
            print(metrics)
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

        # 计算并保存平均指标
        num_images = len(image_list)
        mean_metrics = total_metric / num_images
        f.writelines("Mean metrics,{},{},{},{}\n{},{},{},{},\n,{},{},{},{}".format(
            mean_metrics[0, 0], mean_metrics[0, 1], mean_metrics[0, 2], mean_metrics[0, 3],
            mean_metrics[1, 0], mean_metrics[1, 1], mean_metrics[1, 2], mean_metrics[1, 3],
            mean_metrics[2, 0], mean_metrics[2, 1], mean_metrics[2, 2], mean_metrics[2, 3]
        ))
    print("测试完成，结果已保存")

    # 保存总指标
    # np.savetxt(f"{test_save_path}/{method}_metrics.txt", total_metric)
    #
    # print("测试结束")
    return total_metric/len(image_list)

# import medpy.metric.binary as metric

def calculate_metric_percase(label, label_map):
    def compute_metrics(pred, gt):
        # gt 和 pred 都为 0 的情况
        if np.sum(gt) == 0 and np.sum(pred) == 0:
            dice = 1.0
            ravd = 1.0
            hd = 1.0
            asd = 1.0# 或者对于 hd，可以设为任意较大值，因为无实际意义
        # gt 为 0，pred 不为 0 的情况，跳过计算
        elif np.sum(gt) == 0 and np.sum(pred) != 0:
            return None  # 或者返回其他特殊标志，例如 NaN 或 "skip"
        # gt 不为 0，pred 为 0 的情况，计算为 0
        elif np.sum(gt) != 0 and np.sum(pred) == 0:
            dice = 0.0
            ravd = 0.0
            hd = 0.0
            asd = 0.0
        # 正常情况，gt 和 pred 都不为 0 时，计算指标
        else:
            dice = metric.dc(pred, gt)
            ravd = abs(metric.ravd(pred, gt))
            hd = metric.hd95(pred, gt)
            asd = metric.asd(pred, gt)

        return np.array([dice, ravd,hd,asd])

    # Ground truth binary masks
    et_true = (label == 3).astype(np.float32)
    tc_true = (np.isin(label, [1, 3])).astype(np.float32)
    wt_true = (np.isin(label, [1, 2, 3])).astype(np.float32)

    valid_labels = np.unique(label)

    # Create masks for valid labels in prediction
    et_pred = (label_map == 3).astype(np.float32) if 3 in valid_labels else np.zeros_like(label_map, dtype=np.float32)
    tc_pred = (np.isin(label_map, [1, 3])).astype(np.float32) if any(
        l in valid_labels for l in [1, 3]) else np.zeros_like(label_map, dtype=np.float32)
    wt_pred = (np.isin(label_map, [1, 2, 3])).astype(np.float32) if any(
        l in valid_labels for l in [1, 2, 3]) else np.zeros_like(label_map, dtype=np.float32)

    # Calculate metrics for each class
    et_metrics = compute_metrics(et_pred, et_true)
    tc_metrics = compute_metrics(tc_pred, tc_true)
    wt_metrics = compute_metrics(wt_pred, wt_true)


    return np.array([et_metrics, tc_metrics, wt_metrics])

