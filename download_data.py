import os

from beir import util
from beir.datasets.data_loader import GenericDataLoader

dataset = "dbpedia-entity"
data_path = os.path.join("datasets", dataset)

if not os.path.exists(data_path):
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    print(f"Downloading {dataset}...")
    data_path = util.download_and_unzip(url, "datasets")
    print("Done!")

corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

print(f"Corpus size: {len(corpus)}")
print(f"Queries: {len(queries)}")
print(f"Qrels: {len(qrels)}")

sample_qid = list(queries.keys())[0]
print(f"\nQuery ID: {sample_qid}")
print(f"Query: {queries[sample_qid]}")

sample_did = list(corpus.keys())[0]
doc = corpus[sample_did]
print(f"\nDoc ID: {sample_did}")
print(f"Title: {doc.get('title', 'N/A')}")
print(f"Text: {doc['text'][:200]}...")