import argparse
import json

def update_stats(dataset, uniques):

    if dataset is None:
        return

    for datum in dataset["data"]:

        fs = datum["fs"]
        fs = fs.lower()
        fs = fs.strip()
        uniques.add(fs)

def get_dataset(dataset_filepath):

    try:
        with open(dataset_filepath, "r") as fh:
            return json.load(fh)
    except Exception:

        return None

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=None, type=str,
        required=True)
    parser.add_argument("--total-runs", default=1, type=int)

    args = parser.parse_args()

    DATASETS = ["ksat", "fol", "plogic", "fol_human", "regex"]
    
    stats = {}
    total_count = 0
    for dataset_type in DATASETS:

        stats[dataset_type] = 0
        uniques = set()
        for run_no in range(args.total_runs):
            
            dataset_filepath = "%s/run%d/%s/dataset.json" % (
                args.base_dir,
                run_no,
                dataset_type)
            dataset = get_dataset(dataset_filepath)
            update_stats(dataset, uniques)

        total_count += len(uniques)
        stats[dataset_type] += len(uniques)

    for dataset_type, count in stats.items():
        print(dataset_type, count)

    print("Total:", total_count)
