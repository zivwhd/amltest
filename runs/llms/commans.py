import os

from config.tasks import AGN_TASK, IMDB_TASK, SST_TASK, RTN_TASK, EMOTION_TASK
from config.types_enums import ModelBackboneTypes, AttrScoreFunctions, EvalMetric


def write(_name: str, _line: str):
    f_name = f'submit_{_name}.txt'
    mem = "170g" if "alti" in line else "40g"
    if not os.path.isfile(f_name):
        default_txt = f'''#!/bin/bash

## resource allocation
#SBATCH --job-name=BL_{_name}_
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

#SBATCH --qos=gpu

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem={mem}
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
for model in [ModelBackboneTypes.LLAMA, ModelBackboneTypes.MISTRAL]:
    name += 1
    for attribution_function in [f.deep_lift, f.sequential_integrated_gradients, f.alti, f.input_x_gradient,
                                 f.gradient_shap, f.integrated_gradients, f.llm]:

        for task in [EMOTION_TASK, RTN_TASK, SST_TASK, IMDB_TASK, AGN_TASK]:
            if attribution_function.value in [f.sequential_integrated_gradients.value,
                                              f.alti.value,
                                              f.input_x_gradient.value]:
                name += 1

            metric = "all"
            line = f"python run_baselines.py {task.name} {attribution_function.value} {model.value} {metric}"
            write(name, line)
