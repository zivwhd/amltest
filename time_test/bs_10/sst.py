import sys

sys.path.append("../..")
from config.tasks import SST_TASK
from time_test.run import TestTime

TestTime(task = SST_TASK, beam_size = 10).run()
