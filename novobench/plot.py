from argparse import ArgumentParser
from collections import defaultdict
import os
import sys

from typing import List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt

COLORS=["blue", "orange", "grey"]
def plot_scores(path, title, names, scores, spacing_factor=0.8, figsize=(8, 4)):
    num_compare = len(scores)
    num_items = len(scores[0])
    fig, ax = plt.subplots(figsize=figsize)
    center = sum(i for i in range(num_compare)) / num_compare
    ticks = np.arange(num_items) + center
    for idx, scorelist in enumerate(scores):
        positions = ticks + (idx - 1) * (num_compare - 1) / 2 * spacing_factor
        ax.boxplot(scorelist, positions=positions, color=COLORS[idx], notch=True)
    ax.set_title(title)
    ax.set_xticks(ticks)
    ax.set_xticklabels(names)
    plt.savefig(path, dpi=600)
    plt.close("all")

def read_csv(path):
    with open(path) as f:
        result = {}
        keys = next(f).strip().split(",")
        for line in f:
            values = line.strip().split(",")
            name = values[0]
            if name not in result:
                result[name] = []
            result[name].append({key: value for key, value in zip(keys, values)})
    return result

def reduce_keys(data, ops):
    result = {}
    for name, items in data.items():
        result[name] = {
            k: ops[k]([i[k] for i in items])
            for k in items[0]
        }
    return result

def default_reduce_keys(data):
    return reduce_keys(
        data,
        dict(
            name=lambda x: x[0],
            sequence=lambda x: x[0],
            index=lambda x: x[0],
            sc_rmsd=min,
            sc_tm=max,
            plddt=max,
            ptm=max
        ))

def transpose(data):
    names = sorted([n for n in data])
    result = {
        k: np.array([data[name][k] for name in names])
        for k in data[names[0]]
        if k not in ["name", "sequence", "index"]
    }
    return result

def plot_all_scores(titles, sources):
    result = {}
    for source in sources:
        source_results = transpose(default_reduce_keys(read_csv(source)))
        result

if __name__ == "__main__":
    out_path = sys.argv[1]
    sources = sys.argv[2:]
    sources = [s.split(":") for s in sources]
    os.makedirs(f"{out_path}/plots/", exist_ok=True)
