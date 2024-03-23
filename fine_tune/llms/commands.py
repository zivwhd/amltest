from config.tasks import EMOTION_TASK, AGN_TASK
from config.types_enums import ModelBackboneTypes

for task in [EMOTION_TASK, AGN_TASK]:
    for backbone in [ModelBackboneTypes.LLAMA, ModelBackboneTypes.MISTRAL]:
        for is_bf16 in [True]:
            for is_use_prompt in [True, False]:
                command = f"python run.py --task {task.name} --backbone {backbone.value}"

                if is_bf16:
                    command += " --is_bf16 True"

                if is_use_prompt:
                    command += " --is_use_prompt True"

                print(command)
