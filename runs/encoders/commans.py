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
#SBATCH --mem=30g
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
for model in [ModelBackboneTypes.BERT, ModelBackboneTypes.ROBERTA, ModelBackboneTypes.DISTILBERT]:
    name += 1
    for attribution_function in [f.decompX, f.decompX_class, f.glob_enc, f.deep_lift, f.deep_lift,
                                 f.sequential_integrated_gradients, f.solvability, f.alti, f.input_x_gradient,
                                 f.gradient_shap, f.integrated_gradients, f.lime]:
        if attribution_function.value in [f.sequential_integrated_gradients.value, f.solvability.value, f.alti.value,
                                          f.input_x_gradient.value]:
            name += 1
        for task in [EMOTION_TASK, RTN_TASK, SST_TASK, IMDB_TASK, AGN_TASK]:
            solvability_batch_sizes = [-1]
            metrics_list = ["all"]
            if attribution_function.value == "solvability":
                solvability_batch_sizes = [10, 50, 100]
                metrics_list = [m.value for m in EvalMetric]
            for solvability_batch in solvability_batch_sizes:
                for metric in metrics_list:
                    line = f"python run_baselines.py {task.name} {attribution_function.value} {model.value} {metric} {solvability_batch}"
                    write(name, line)
