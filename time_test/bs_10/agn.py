import sys

sys.path.append("../..")
from config.tasks import AGN_TASK
from time_test.run import TestTime

TestTime(task = AGN_TASK, beam_size = 10).run()
