from flask import Flask, request
from functools import lru_cache
import math
from colbert import Searcher
from argparse import ArgumentParser


parser = ArgumentParser(
    prog='ColBERT Server',
    description='Start a ColBERT server.'
)
parser.add_argument(
    '-n', '--indexname', 
    type=str, 
    default="index_fringe",
    help='index name'
)
parser.add_argument(
    '-r', '--indexroot', 
    type=str, 
    default="experiments/exp_fringe/indexes",
    help='index root'
)
parser.add_argument(
    '-a', '--address', 
    type=str, 
    default="0.0.0.0",
    help='host ip'
)
parser.add_argument(
    '-p', '--port', 
    type=int, 
    default=8001,
    help='server port'
)
args = parser.parse_args()


app = Flask(__name__)

searcher = Searcher(index=args.indexname, index_root=args.indexroot)
counter = {"api" : 0}

@lru_cache(maxsize=1000000)
def api_search_query(query, k):
    print(f"Query={query}")
    if k == None: k = 10
    k = min(int(k), 100)
    pids, ranks, scores = searcher.search(query, k=100)
    pids, ranks, scores = pids[:k], ranks[:k], scores[:k]
    passages = [searcher.collection[pid] for pid in pids]
    probs = [math.exp(score) for score in scores]
    probs = [prob / sum(probs) for prob in probs]
    topk = []
    for pid, rank, score, prob in zip(pids, ranks, scores, probs):
        text = searcher.collection[pid]            
        d = {
            'text': text, 
            'pid': pid, 
            'rank': rank, 
            'score': score, 
            'prob': prob
        }
        topk.append(d)
    topk = list(sorted(topk, key=lambda p: (-1 * p['score'], p['pid'])))
    return {"query" : query, "topk": topk}

@app.route("/api/search", methods=["GET"])
def api_search():
    if request.method == "GET":
        counter["api"] += 1
        print("API request count:", counter["api"])
        return api_search_query(
            request.args.get("query"), 
            request.args.get("k")
        )
    else:
        return ('', 405)

if __name__ == "__main__":
    app.run(args.address, args.port)
