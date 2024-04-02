import sys

sys.path.append("../..")
from config.tasks import IMDB_TASK
from time_test.run import TestTime

TestTime(task = IMDB_TASK, beam_size = 20).run()
