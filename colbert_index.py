import os
os.environ['CUDA_HOME'] = os.environ['CONDA_PREFIX']
os.environ['LIBRARY_PATH'] = os.environ['CONDA_PREFIX']+"/lib"
os.environ['LD_LIBRARY_PATH'] = os.environ['CONDA_PREFIX']+"/lib"

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Collection
from colbert import Indexer
from argparse import ArgumentParser


if __name__ == "__main__":

    parser = ArgumentParser(
        prog='ColBERT Indexing',
        description='Indexing a collection of text.'
    )

    parser.add_argument(
        '-b', '--nbits', 
        type=int, 
        default=2,
        help='each dimension encoding bits'
    )

    parser.add_argument(
        '-l', '--maxlen', 
        type=int, 
        default=300,
        help='max number of tokens per document'
    )

    parser.add_argument(
        '-c', '--checkpoint', 
        type=str, 
        default="colbertv2.0",
        help='trained checkpoint to use'
    )

    parser.add_argument(
        '-p', '--path', 
        type=str, 
        default="fringe_collection.tsv",
        help='collection path'
    )

    parser.add_argument(
        '-i', '--index', 
        type=str, 
        default="index_fringe",
        help='new index name'
    )

    parser.add_argument(
        '-e', '--experiment', 
        type=str, 
        default="exp_fringe",
        help='experiment name'
    )

    parser.add_argument(
        '-r', '--ranks', 
        type=int, 
        default=1,
        help='number of GPUs to use'
    )

    parser.add_argument(
        '-s', '--bsize', 
        type=int, 
        default=16,
        help='batch size'
    )

    args = parser.parse_args()
    

    collection = Collection(path=args.path)

    with Run().context(
        RunConfig(nranks=args.ranks, experiment=args.experiment)
    ):  
        config = ColBERTConfig(
            doc_maxlen=args.maxlen, 
            nbits=args.nbits, 
            index_bsize=args.bsize
        )
        indexer = Indexer(
            checkpoint=args.checkpoint, 
            config=config
        )
        indexer.index(
            name=args.index, 
            collection=collection, 
            overwrite=True
        )

    print("Index created at: " + indexer.get_index())