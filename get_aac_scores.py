import pandas as pd 
import os 
import json 
from tqdm import tqdm 
from itertools import chain 
import collections 
import seaborn as sns 
import matplotlib.pyplot as plt
from tabulate import tabulate
import argparse



tqdm.pandas()
sns.set_theme(style='darkgrid', font_scale=1.1)

def get_cov_score(row, length):
    scores = []
    for i in range(length):
        score = len(row[i]) / len(row["alignment"]) if len(row["alignment"]) > 0 else 1
        scores.append(score)
    return pd.Series(scores)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--nmax", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_json(os.path.join(args.data_dir, "source_coverage.jsonl"), lines=True)
    df_coverage_score = df.apply(lambda row: get_cov_score(row, args.nmax), axis=1)
    
    scores = (1 - df_coverage_score).sum(axis=1).divide(args.nmax)
    aac, std = scores.mean(), scores.std()

    print(f"AAC: {aac * 100} ± {std * 100}")

    p = sns.lineplot(data=df_coverage_score.mean(axis=0).to_frame(name=os.path.basename(args.data_dir)))
    sns.move_legend(p, "lower right")
    p.set_ylim(0.3, 1.05)
    p.set_xlabel("Number of source articles")
    p.set_ylabel("f(x) -- Prop")

    figure = p.get_figure()
    figure.savefig(os.path.join(args.data_dir, "plot.png"), bbox_inches='tight')
    plt.close()