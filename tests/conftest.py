import os, sys
import yaml

ROOT_DIR = None
with open('../cfg/project/default.yaml', 'r') as f: ROOT_DIR = yaml.load(f, Loader=yaml.FullLoader)['root_dir']
sys.path.insert(0, os.path.abspath(f"{ROOT_DIR}/src"))
