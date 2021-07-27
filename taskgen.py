import argparse
import copy
import logging
import os
import shlex
import time
from datetime import datetime

import torch
from sklearn.model_selection import ParameterGrid
import subprocess
import json

import numpy as np

from modules.csv_utils_2 import CsvUtils2

parser = argparse.ArgumentParser('Generic linux task generator', add_help=False)

parser.add_argument('--sequence-name', default='sequence', type=str, help='sequence name')
parser.add_argument('--template', default='template_loc.sh', type=str)
parser.add_argument('--script', default='main.py', type=str)
parser.add_argument('--num-repeat', default=1, type=int,
                    help='how many times each set of parameters should be repeated for testing stability')
parser.add_argument('--num-tasks-in-parallel', default=6, type=int)
parser.add_argument('--num-cuda-devices-per-task', default=1, type=int)
parser.add_argument('--is-single-task', action='store_true')
parser.add_argument('--is-force-start', action='store_false')


class DictToObj:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def add_other_args(args, args_other):
    args = args.__dict__

    arg_name = None
    arg_params = []

    def handle_args(args, arg_name, arg_params):
        if arg_name is not None and len(arg_params):
            # check by type, int, float, bool, str
            is_parsed = False
            try:
                args[arg_name] = [int(it) for it in arg_params]
                is_parsed = True
            except ValueError:
                pass

            if not is_parsed:
                try:
                    args[arg_name] = [float(it) for it in arg_params]
                    is_parsed = True
                except ValueError:
                    pass

            if not is_parsed:
                try:
                    for it in arg_params:
                        if it.lower() != 'false' and it.lower() != 'true':
                            raise ValueError
                    args[arg_name] = [it.lower() == 'true' for it in arg_params]
                    is_parsed = True
                except ValueError:
                    pass

            if not is_parsed:
                args[arg_name] = arg_params
            if isinstance(args[arg_name], list):
                if len(args[arg_name]) == 1:
                    args[arg_name] = args[arg_name][0]
        elif arg_name is not None:
            args[arg_name] = ""

    for each in args_other:
        if each.startswith('-'):
            handle_args(args, arg_name, arg_params)
            arg_params = []
            arg_name = each.strip()
        else:
            if arg_name is not None:
                arg_params.append(each.strip())

    handle_args(args, arg_name, arg_params)
    return DictToObj(**args)


args, args_other = parser.parse_known_args()
args = add_other_args(args, args_other)
args.sequence_name_orig = args.sequence_name
args.sequence_name += ('-' + datetime.utcnow().strftime(f'%y-%m-%d-%H-%M-%S'))


def create_dir(dirPath):
    if not os.path.exists(dirPath):
        try:
            os.makedirs(dirPath)
        except Exception as e:
            print(e)


def write_text_file(filepath, text, encoding='utf-8'):
    try:
        with open(filepath, 'w', encoding=encoding, errors="ignore") as fp:
            fp.write(text)
    except Exception as e:
        print(e)


def read_all_as_string(path, encoding=None):
    result = None
    if os.path.exists(path):
        with open(path, 'r', encoding=encoding) as fp:
            result = ''.join(fp.readlines())
    return result


def wait_processes(procs, wait_for, cuda_devices):
    while len(procs.keys()) >= wait_for:
        time.sleep(1)
        filtered = {}
        for id, proc in procs.items():
            if proc.poll() is not None:
                if proc.stdout:
                    logging.info(str(id) + proc.stdout)
                if proc.stderr:
                    logging.error(str(id) + proc.stderr)
                if proc.returncode:
                    logging.info("Process %d exited with code: %d" % (id, proc.returncode))
                for dev_id in proc.cuda_devices_for_run:
                    if dev_id in cuda_devices:
                        cuda_devices.remove(dev_id)
            else:
                filtered[id] = proc
        procs = filtered
    return procs


create_dir('./results')
path_sequence = f'./results/{args.sequence_name}'
path_sequence_scripts = f'{path_sequence}/scripts'
create_dir(path_sequence)
create_dir(path_sequence_scripts)
create_dir('./logs')
create_dir('./artefacts')

rootLogger = logging.getLogger()
logFormatter = logging.Formatter("%(asctime)s [%(process)d] [%(thread)d] [%(levelname)s]  %(message)s")
rootLogger.level = logging.DEBUG  # level

fileHandler = logging.FileHandler(f'{path_sequence}/log.txt')
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

# Skip these
if hasattr(args, 'tf_path_test'):
    if isinstance(args.tf_path_test, list):
        args.tf_path_test = ' '.join(args.tf_path_test)
    if isinstance(args.tf_path_train, list):
        args.tf_path_train = ' '.join(args.tf_path_train)

args_with_multiple_values = {}
single_args = []
for key, value in args.__dict__.items():
    if isinstance(value, list):
        if len(value) > 1:
            args_with_multiple_values[key] = value
    elif value not in args.__dict__.keys() and value not in args.__dict__.values():
        single_args.append(value)

grid_runs = list(ParameterGrid(args_with_multiple_values))

runs = []
for grid_each in grid_runs:
    for _ in range(args.num_repeat):
        run = copy.deepcopy(args.__dict__)
        run.update(grid_each)
        runs.append(run)

if len(runs) == 0:
    logging.error('no grid search combinations found')
    exit()

logging.info(f'planned runs: {len(runs)}')
logging.info(f'grid_runs:\n{json.dumps(grid_runs, indent=4)}')

if not args.is_force_start:
    print('are tests ok? proceed?')
    if input('[y/n]: ') != 'y':
        exit()

CsvUtils2.create_global(path_sequence)

max_cuda_devices = 0
cuda_devices_available = 0
if not torch.cuda.is_available():
    args.device = 'cpu'
    logging.info('CUDA NOT AVAILABLE')
else:
    max_cuda_devices = torch.cuda.device_count()
    cuda_devices_available = np.arange(max_cuda_devices).astype(np.int).tolist()

cuda_devices_in_use = []
parallel_processes = {}

idx_cuda_device_seq = 0
cuda_devices_list = np.arange(0, max_cuda_devices, dtype=np.int).tolist()

for idx_run, run in enumerate(runs):
    cmd_params = []
    for key, value in run.items():
        if key.startswith('-'):
            if value is not None:
                cmd_params.append(key + ' ' + str(value))
            else:
                cmd_params.append(key)

    str_cmd_params = ' '.join(cmd_params)

    str_cuda = ''
    cuda_devices_for_run = []
    if max_cuda_devices > 0:
        if args.num_tasks_in_parallel <= args.num_cuda_devices_per_task:
            for device_id in cuda_devices_available:
                if device_id not in cuda_devices_in_use:
                    cuda_devices_for_run.append(device_id)
                    if len(cuda_devices_for_run) >= args.num_cuda_devices_per_task:
                        break

            if len(cuda_devices_for_run) < args.num_cuda_devices_per_task:
                # reuse existing devices #TODO check to reuse by least recent device
                for device_id in cuda_devices_in_use:
                    cuda_devices_for_run.append(device_id)
                    if len(cuda_devices_for_run) >= args.num_cuda_devices_per_task:
                        break

            for device_id in cuda_devices_for_run:
                if device_id not in cuda_devices_in_use:
                    cuda_devices_in_use.append(device_id)
        else:
            while len(cuda_devices_for_run) < args.num_cuda_devices_per_task:
                cuda_devices_for_run.append(cuda_devices_list[idx_cuda_device_seq])
                idx_cuda_device_seq += 1
                if idx_cuda_device_seq >= len(cuda_devices_list):
                    idx_cuda_device_seq = 0

        if len(cuda_devices_for_run):
            str_cuda = f'CUDA_VISIBLE_DEVICES={",".join([str(it) for it in cuda_devices_for_run])} '

    # Detect HPC
    if '/mnt/home/' in os.getcwd():
        str_cuda = ''

    run_time = datetime.utcnow().strftime(f'%y-%m-%d--%H-%M-%S')
    run_name = args.sequence_name_orig + f"-{idx_run + 1}-run-" + run_time
    path_run_sh = f'{path_sequence_scripts}/{run_name}.sh'
    cmd = f'{str_cuda}python ./{args.script} --id {idx_run + 1} --run-name {run_name}' \
          f' {str_cmd_params}' \
          f' --sequence-name {args.sequence_name_orig}'
    print(path_run_sh)
    print(cmd)
    write_text_file(
        path_run_sh,
        read_all_as_string(args.template) +
        f'\n{cmd} > ./logs/{run_name}.log 2>&1'
    )

    cmd = f'chmod +x {path_run_sh}'
    stdout = subprocess.call(shlex.split(cmd))

    logging.info(f'{idx_run}/{len(runs)}: {path_run_sh}\n{cmd}')
    process = subprocess.Popen(path_run_sh, shell=False)
    process.cuda_devices_for_run = cuda_devices_for_run
    parallel_processes[idx_run] = process

    time.sleep(1.1)  # delay for timestamp based naming

    parallel_processes = wait_processes(parallel_processes, args.num_tasks_in_parallel, cuda_devices_in_use)

    if args.is_single_task:
        logging.info('Single task test debug mode completed')
        exit()

wait_processes(parallel_processes, 1, cuda_devices_in_use)

logging.info('TaskGen finished')
