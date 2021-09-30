import subprocess
import os
from sys import platform


def run_experiment(experiment_params, main_program_name):

    # & for windows or ; for mac
    if os.name == 'nt':
        separate = '&'
    else:
        separate = ';'

    command = f'python {main_program_name}.py '

    for param in list(experiment_params.keys()):
        command += f' --{param} {experiment_params[param]} '

    process = subprocess.Popen(command, shell=True)

    return process