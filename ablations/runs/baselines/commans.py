import os

from config.tasks import AGN_TASK, IMDB_TASK, SST_TASK, RTN_TASK, EMOTION_TASK
from config.types_enums import ModelBackboneTypes, AttrScoreFunctions, EvalMetric


def write(_name: str, _line: str):
    f_name = f'submit_{_name}.txt'
    if not os.path.isfile(f_name):
        default_txt = f'''#!/bin/bash

## resource allocation
#SBATCH --job-name=BL_{_name}_
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

#SBATCH --qos=gpu

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=25g
#SBATCH --gpus=1
#SBATCH --nodelist=gpu7

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
for model in [ModelBackboneTypes.ROBERTA, ModelBackboneTypes.BERT]:
    for attribution_function in [f.sequential_integrated_gradients, f.integrated_gradients]:
        for task in [EMOTION_TASK, RTN_TASK, SST_TASK]:
            metrics_list = ["all"]
            for metric in metrics_list:
                line = f"python run_baselines.py {task.name} {attribution_function.value} {model.value} {metric}"
                print(line)
                write(name, line)
                name += 1
