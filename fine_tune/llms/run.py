import sys

sys.path.append("../..")

from fine_tune.llms.fine_tune_llama import LlmFineTune
from config.tasks import EMOTION_TASK

LlmFineTune(task = EMOTION_TASK, is_bf16 = True, is_use_prompt = True)
