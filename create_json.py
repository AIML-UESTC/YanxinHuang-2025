import json
import os
data_dir = r"images"
label_dir = r"1st_manual"
# MTG_dir =
# Unet_dir =

data_files = os.listdir(data_dir)
label_files = os.listdir(label_dir)

data_dict = {}
for data_file in data_files:
    data_key = os.path.splitext(data_file)[0][:2]
    # print(data_key)
    # print(label_files)
    for label_file in label_files:
        label_file1 = label_file[:2]
        # print(label_file1)
        if data_key in label_file1:
            label_key = os.path.splitext(label_file)[0]
            data_dict[data_key] = {'data_path': os.path.join(data_dir, data_file),
                                   'label_path': os.path.join(label_dir, label_file)}
            break
with open('output.json', 'w') as f:
    json.dump(data_dict, f,indent=4)
