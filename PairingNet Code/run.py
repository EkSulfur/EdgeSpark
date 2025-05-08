import os
import subprocess
path = './'
run_file = path + "PairingNet_train_val_test.py"
wsl_python_interpreter = "/home/eksulfur/miniconda3/envs/PairingNet/bin/python3"

# 直接使用 WSL 内部的 Python 解释器路径
env = os.environ.copy()

# subprocess.run([wsl_python_interpreter, run_file, "--model_type=matching_train", "--epoch=2"], env=env)
subprocess.run([wsl_python_interpreter, run_file, "--model_type=matching_test"], env=env)
# subprocess.run([wsl_python_interpreter, run_file, "--model_type=save_stage1_feature"], env=env)

# cmd = '{} -m torch.distributed.launch --nproc_per_node 4 PairingNet_train_val_test.py --model_type=searching_train'.format(wsl_python_interpreter)
# subprocess.run(cmd, shell=True, cwd=path) # 注意：如果使用 shell=True，路径和命令构建方式可能需要调整
# cmd = '{} texture_countour_double_GCN.py --model_type=searching_test'.format(wsl_python_interpreter)
# subprocess.run(cmd, shell=True, cwd=path)

# subprocess.run(["python", run_file, "--model_type=matching_train"])
# subprocess.run(["python", run_file, "--model_type=matching_test"])
# subprocess.run(["python", run_file, "--model_type=save_stage1_feature"])
# cmd = 'python -m torch.distributed.launch --nproc_per_node 4 PairingNet_train_val_test.py --model_type=searching_train'
# subprocess.run(cmd, shell=True, cwd=path)
# cmd = 'python texture_countour_double_GCN.py --model_type=searching_test'
# subprocess.run(cmd, shell=True, cwd=path)