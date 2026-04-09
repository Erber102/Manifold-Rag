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
    
    corpus, test_queries, test_qrels = GenericDataLoader(data_path).load(split="test")
    _, train_queries, train_qrels = GenericDataLoader(data_path).load(split="train")
    print(f"\n{dataset}:")
    print(f"  Corpus: {len(corpus)} docs")
    print(f"  Train queries: {len(train_queries)} | train qrels: {len(train_qrels)}")
    print(f"  Test queries:  {len(test_queries)} | test qrels:  {len(test_qrels)}")
    overlap = set(train_queries.keys()) & set(test_queries.keys())
    print(f"  Train/test query id overlap: {len(overlap)} (should be 0)")