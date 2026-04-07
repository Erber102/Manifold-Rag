import os
from beir import util
from beir.datasets.data_loader import GenericDataLoader

datasets = ["scifact", "fiqa"]

for dataset in datasets:
    data_path = os.path.join("data", dataset)
    if not os.path.exists(data_path):
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        print(f"Downloading {dataset}...")
        util.download_and_unzip(url, "data")
        print(f"{dataset} done!")
    
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    print(f"\n{dataset}:")
    print(f"  Corpus: {len(corpus)} docs")
    print(f"  Queries: {len(queries)}")
    print(f"  Qrels: {len(qrels)}")