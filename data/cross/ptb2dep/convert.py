from StanfordDependencies import StanfordDependencies
sd = StanfordDependencies('/clwork/zchen/checkpoints/UniTP/006/data')
corp_path = '/cldata/LDC/penn_treebank_3/treebank_3/parsed/mrg/wsj'
from nltk.corpus import BracketParseCorpusReader
reader = BracketParseCorpusReader(corp_path, r".*/wsj_.*\.mrg")

# from nltk.corpus import treebank
from time import time
from tqdm import tqdm

def proc_batch(str_trees):
    start = time()
    sents = sd.convert_trees(str_trees)#, **conversion_args)
    return f'{len(str_trees) / (time() - start):.1f} sents/sec.'

trees = reader.parsed_sents()
sd.convert_corpus(trees)
# n = 50
# batch = None
# with tqdm(total = len(trees)) as qbar:
#     for x in trees:
#         if batch is None:
#             batch = []
#         batch.append(x)
#         if len(batch) == n:
#             qbar.desc = proc_batch(batch)
#             batch = None
#         qbar.update(1)

#     if batch:
#         qbar.desc = proc_batch(batch)