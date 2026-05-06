"""
@author : Hyunwoong
@when : 2019-12-22
@homepage : https://github.com/gusdnd852
"""
import math
from collections import Counter
import numpy as np


def bleu_stats(hypothesis, reference):
    stats = [len(hypothesis), len(reference)]
    for n in range(1, 5):
        s_ngrams = Counter([tuple(hypothesis[i:i+n]) for i in range(len(hypothesis)+1-n)])
        r_ngrams = Counter([tuple(reference[i:i+n])  for i in range(len(reference)+1-n)])
        stats.append(max(sum((s_ngrams & r_ngrams).values()), 0))
        stats.append(max(len(hypothesis)+1-n, 0))
    return stats


def bleu(stats):
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0
    c, r = stats[:2]
    log_bleu_prec = sum(
        math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])
    ) / 4.
    return math.exp(min(0, 1 - float(r) / c) + log_bleu_prec)


def get_bleu(hypotheses, reference):
    stats = np.array([0.] * 10)
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    return 100 * bleu(stats)


def idx_to_word(indices, vocab):
    special = {vocab.stoi[s] for s in ['<pad>', '<sos>', '<eos>', '<unk>']}
    return ' '.join(vocab.itos.get(i.item() if hasattr(i, 'item') else i, '')
                    for i in indices
                    if (i.item() if hasattr(i, 'item') else i) not in special)
