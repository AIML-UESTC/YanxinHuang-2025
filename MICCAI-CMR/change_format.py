import numpy as np
from scipy.io import loadmat, savemat
import os


# def sos(img, axis):
#     return np.sqrt(np.sum(np.abs(img) ** 2, axis=axis))


# def crop(img, new_shape):
#     x_center, y_center = img.shape[0] // 2, img.shape[1] // 2
#     new_x, new_y = new_shape[:2]
#     cropped_img = img[x_center - new_x // 2: x_center + new_x // 2,
#                   y_center - new_y // 2: y_center + new_y // 2]
#     if len(new_shape) > 2:
#         cropped_img = cropped_img[:, :, :new_shape[2]]
#     if len(new_shape) > 3:
#         cropped_img = cropped_img[:, :, :, :new_shape[3]]
#     return cropped_img


def crop(img, new_shape):
    if isinstance(new_shape, int):
        new_shape = [new_shape]

    m = img.shape
    s = list(new_shape)

    if len(s) < len(m):
        s += [1] * (len(m) - len(s))

    if all(mi == si for mi, si in zip(m, s)):
        return img

    idx = []
    for n in range(len(s)):
        center = m[n] // 2
        start = center - (s[n] // 2)
        end = start + s[n]
        idx.append(slice(start, end))

    return img[tuple(idx)]

def run4Ranking(img, filetype):
    isBlackBlood = 'blackblood' in filetype.lower()
    isMapping = 't1map' in filetype.lower() or 't2map' in filetype.lower()

    # if isBlackBlood:
    #     sx, sy, sz = img.shape
    #
    # else:
    sx, sy, sz, t = img.shape

    if sz < 3:
        sliceToUse = list(range(sz))
    else:
        # sliceToUse = [sz // 2 - 1, sz // 2]
        sliceToUse = [round(sz / 2)- 1, round(sz / 2)]

    if isBlackBlood:
        timeFrameToUse = [0]
    elif isMapping:
        timeFrameToUse = list(range(t))
    else:
        timeFrameToUse = list(range(3))

    # sosImg = np.squeeze(sos(img, axis=2))

    sosImg = img
    #
    # if sz == 1:
    #     sosImg = sosImg.reshape((sosImg.shape[0], sosImg.shape[1], 1, sosImg.shape[2]))

    if isBlackBlood:
        selectedImg = sosImg[:, :, sliceToUse]
        img4ranking = crop(np.abs(selectedImg), (round(sx / 3), round(sy / 2), len(sliceToUse))).astype(np.float32)
    else:
        selectedImg = sosImg[:, :, sliceToUse, :len(timeFrameToUse)]
        img4ranking = crop(np.abs(selectedImg), (round(sx / 3), round(sy / 2), len(sliceToUse), len(timeFrameToUse))).astype(
            np.float32)

    return img4ranking


# 示例代码：遍历目录，处理文件并保存结果
def process_directory(base_path, main_save_path, task_type, data_type_list, set_name):
    coil_info = 'MultiCoil'

    for data_type in data_type_list:
        task_dir = os.path.join(base_path,data_type, set_name, task_type)
        for root, dirs, files in os.walk(task_dir):
            for file_name in files:
                if not file_name.endswith('.mat'):
                    continue

                full_ks_file_path = os.path.join(root, file_name)
                kspace_data = loadmat(full_ks_file_path)
                img = kspace_data['img4ranking']

                # img = np.fft.ifft2(kspace)
                img4ranking = run4Ranking(img, file_name)

                save_dir = os.path.join(main_save_path, coil_info, data_type, set_name, task_type,
                                        os.path.basename(root))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                save_path = os.path.join(save_dir, file_name)
                savemat(save_path, {'img4ranking': img4ranking})

                print(f"{data_type} multicoil data generation successful!")


# 使用示例
base_path = '/home/qitam/sdb2/home/qiteam_project/huang/PromptMR-main/cmr_challenge_results/reproduce_promptmr_12_cascades_cmrxrecon/Task2'
main_save_path = '/home/qitam/sdb2/home/qiteam_project/huang/PromptMR-main/cmr_challenge_results/Task2_change_1'
task_type = 'Task2'
# data_type_list = ['Aorta', 'BlackBlood', 'Cine', 'Flow2d', 'Mapping', 'Tagging']
data_type_list = ['Aorta',  'Cine', 'Mapping', 'Tagging']
# data_type_list = ['Cine']
set_name = 'ValidationSet'

process_directory(base_path, main_save_path, task_type, data_type_list, set_name)
