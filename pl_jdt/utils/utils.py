import os, random, string
from datetime import datetime

# pytoch
import pl_jdt.utils.transforms as T

# def get_transforms(mode):
#     if mode == 'train':
#         return transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])
#     return transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])

def get_exp_path_and_run_name(output_path,exp_name):
    random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
    today = datetime.today().strftime("%Y-%m-%d")
    curr_time = datetime.today().strftime("%H-%M")
    run_name = f"{curr_time}-{exp_name}-{random_str}"
    exp_path = f'{output_path}/{today}/{run_name}'
    return exp_path, run_name, today

def create_log_dir(output_path, exp_path, today):
    if not os.path.exists(output_path): os.makedirs(output_path)
    if not os.path.exists(f'{output_path}/{today}'): os.makedirs(f'{output_path}/{today}')
    if not os.path.exists(exp_path): 
        os.makedirs(exp_path)
        os.makedirs(f"{exp_path}/ckpts")
        # os.makedirs(f"{exp_path}/optims")
        os.makedirs(f"{exp_path}/figs")
        # os.makedirs(f"{exp_path}/logs")
    else:
        raise Exception(f"Experiment path {exp_path} already exists")

def exp(output_path, exp_name):
    exp_path, run_name, today = get_exp_path_and_run_name(output_path, exp_name)
    create_log_dir(output_path, exp_path, today)
    return exp_path, run_name