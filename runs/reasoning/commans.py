import os

from config.tasks import IMDB_TASK, SST_TASK
from config.types_enums import ModelBackboneTypes, AttrScoreFunctions


def write(_name: str, _line: str):
    f_name = f'submit_{_name}.txt'
    if not os.path.isfile(f_name):
        default_txt = f'''#!/bin/bash

## resource allocation
#SBATCH --job-name=REASONING_BL_{_name}_
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

#SBATCH --qos=gpu

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=25g
#SBATCH --gres=gpu:a100:1


#SBATCH --no-requeue


## modules and apps
module load anaconda3
source activate NEEA_2

## run
'''
        with open(f_name, 'a') as file:
            file.write(default_txt)
    with open(f_name, 'a') as file:
        file.write(_line + '\n')


name = 0
f = AttrScoreFunctions
for task in [SST_TASK, IMDB_TASK]:
    for model in [ModelBackboneTypes.BERT]:
        name += 1
        for attribution_function in [f.solvability]:
            for i in range(10):
                line = f"python run_baselines.py {task.name} {attribution_function.value} {model.value} {i * 20} {(i + 1) * 20}"
                print(line)
                write(name, line)
                name += 1
