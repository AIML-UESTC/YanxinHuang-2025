# import shutil
# import os
#
# # 指定要移动的文件夹路径
# source_folder = '/home/qitam/sdb2/home/qiteam_project/huang/PromptMR-main/cmr_challenge_results/reproduce_promptmr_12_cascades_cmrxrecon/Submission/Aorta/ValidationSet/Task2'
# # destination_path = '/home/qitam/sdb2/home/qiteam_project/huang/PromptMR-main/cmr_challenge_results/reproduce_promptmr_12_cascades_cmrxrecon/Submission'
#
# # 指定目标路径
# destination_path = '/home/qitam/sdb2/home/qiteam_project/huang/PromptMR-main/cmr_challenge_results/reproduce_promptmr_12_cascades_cmrxrecon/Task2/Aorta/ValidationSet'
#
# # source_folder = '/home/qitam/sdb2/home/qiteam_project/huang/PromptMR-main/cmr_challenge_results/reproduce_promptmr_12_cascades_cmrxrecon/Task2/Tagging'
#
# # 确保目标路径存在，如果不存在则创建
# os.makedirs(destination_path, exist_ok=True)
#
# # 移动文件夹
# shutil.move(source_folder, destination_path)
#
# print(f"文件夹已从 {source_folder} 移动到 {destination_path}")
import synapseclient
from synapseclient import File
syn = synapseclient.login()

# Add a local file to an existing project (syn12345) on Synapse
file = File(path='/home/qitam/sdb2/home/qiteam_project/huang/PromptMR-main/cmr_challenge_results/Task2_change_1/Task2.zip', parent='syn60281221')
file = syn.store(file)