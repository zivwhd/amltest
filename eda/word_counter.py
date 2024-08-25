from datasets import load_dataset

from config.tasks import EMOTION_TASK, SST_TASK, IMDB_TASK, RTN_TASK, AGN_TASK

for task in [EMOTION_TASK, SST_TASK, IMDB_TASK, RTN_TASK, AGN_TASK]:
    ds = load_dataset(task.dataset_name)

    words_counter = 0
    items_counter = 0
    for k in ds.keys():
        for item in ds[k]:
            words_counter += len(item[task.dataset_column_text].split())
            items_counter += 1
    print(f"{task.name} - {words_counter / items_counter:.1f}")
