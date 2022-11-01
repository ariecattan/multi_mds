import os 
import argparse
import pandas as pd 
import numpy as np
from datasets import load_dataset
import logging
from tqdm import tqdm 
from itertools import chain 
import json 
import collections 

tqdm.pandas()

def read_dir_to_np(dir_results, df_oie):
    '''
    :param args:
    :return: np array with 5 column : [scores, index_sum, index_doc, topic_id, story_id]
    '''
    npys = os.listdir(dir_results)
    list_index_doc = sorted([file for file in npys if 'index_doc' in file])
    list_index_sum = sorted([file for file in npys if 'index_sum' in file])
    list_topic_id = sorted([file for file in npys if 'topic_id' in file])
    list_story_id = sorted([file for file in npys if 'story_id' in file])
    list_scores = sorted([file for file in npys if 'scores' in file])
    
    scores = np.concatenate(
        [np.load(os.path.join(dir_results, score_npy)) for score_npy in list_scores])
    index_sum = np.concatenate(
        [np.load(os.path.join(dir_results, index_sum_npy)) for index_sum_npy in list_index_sum])
    index_doc_npy = np.concatenate(
        [np.load(os.path.join(dir_results, index_doc_npy)) for index_doc_npy in list_index_doc])
    topic_id = np.concatenate(
        [np.load(os.path.join(dir_results, topic_npy)) for topic_npy in list_topic_id])
    # stories_id = np.concatenate(
    #     [np.load(os.path.join(args.dir_results, story_npy)) for story_npy in list_story_id])
    stories_id = np.array(df_oie.iloc[index_doc_npy]['story_id'])

    result = np.column_stack((scores, index_sum, index_doc_npy, topic_id, stories_id))

    logger.info(f'number of pairs: {result.shape}')
    return result


def flatten(lst):
    return set(chain.from_iterable(lst))



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    root_logger = logging.getLogger()
    logger = root_logger.getChild(__name__)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt="%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    with open(os.path.join(args.data_dir, "num_of_pairs.txt"), "r") as f:
        num_of_pairs = int(f.readlines()[0])

    logger.info("Loading data..")
    df_oie = pd.read_csv(os.path.join(args.data_dir, "df_oie.csv"))
    if "df_pairs.parquet" in os.listdir(args.data_dir):
        pairs = pd.read_parquet(os.path.join(args.data_dir, "df_pairs.parquet"))
    else:
        scores = read_dir_to_np(os.path.join(args.data_dir, "result_npy"), df_oie)
        pairs = pd.DataFrame(scores, columns=["score", "index_sum", "index_doc", "topic_id", "story_id"])
        pairs[['index_sum', 'index_doc', 'topic_id', 'story_id']] = pairs[['index_sum', 'index_doc', 'topic_id', 'story_id']].astype(int)
        pairs.to_parquet(os.path.join(args.data_dir, "df_pairs.parquet"))
    pairs['aligned'] = pairs['score'] > args.threshold
    
    logger.info(f"num of computed pairs: {len(pairs)}")
    logger.info(f"num of original pairs: {num_of_pairs}")
    assert len(pairs) == num_of_pairs
    
    if "df_alignments.parquet" in os.listdir(args.data_dir):
        alignments = pd.read_parquet(os.path.join(args.data_dir, "df_alignments.parquet"))
    else:
        logger.info("create dataframe of alignments")
        data = []
        for (topic, story_id), row in tqdm(pairs.groupby(['topic_id', 'story_id'])):
            oie_sum_group = row.groupby('index_sum')[['aligned']].apply(sum) > 0
            data.append((topic, story_id, oie_sum_group[oie_sum_group['aligned'] == True].index.values.tolist()))
        alignments = pd.DataFrame(data, columns=['topic', 'story', 'alignment'])
        alignments.to_parquet(os.path.join(args.data_dir, "df_alignments.parquet"))


    logger.info("create topic df..")
    all_oie = pd.DataFrame(pairs.groupby('topic_id')['index_sum'].progress_apply(set))
    total_coverage = pd.DataFrame(alignments.groupby('topic')['alignment'].progress_apply(flatten))
    topics = pd.concat([all_oie, total_coverage], axis=1)
    topics.rename(columns={'index_sum': 'all_oie'}, inplace=True)
    topics['non_aligned'] = topics['all_oie'] - topics['alignment']
    topics['total_coverage'] = topics['alignment'].apply(len) / topics['all_oie'].apply(len)
    topics['hallucinated'] = topics['non_aligned'].apply(len) / topics['all_oie'].apply(len)
    

    alignments['length'] = alignments['alignment'].apply(len)
    max_idx_per_topic = alignments.groupby('topic')[['length']].idxmax()
    topics['dominant'] = alignments.loc[max_idx_per_topic['length']]['alignment'].apply(set).tolist()
    topics["relative_dominant_score"] = topics["dominant"].apply(len) / topics["alignment"].apply(len)
    topics["absolute_dominant_score"] = topics["dominant"].apply(len) / topics["all_oie"].apply(len)


    logger.info("Create df of all alignments")
    all_coverage_data = collections.defaultdict(list)
    data_stories = collections.defaultdict(list)
    alignments["alignment"] = alignments["alignment"].apply(set)

    for topic_id, group in tqdm(alignments.groupby("topic")[["story", "alignment"]]):
        topic_indices = topics.loc[topic_id]["all_oie"].copy()
        total_aligned = set()
        while len(group) > 0:
            group["diff"] = topic_indices - group["alignment"]
            group["diff_len"] = group["diff"].apply(len)
            idx = group["diff_len"].argmin()
            row_id = group.index[idx]
            to_add = group.loc[row_id]["alignment"]
            total_aligned.update(to_add)
            all_coverage_data[topic_id].append(total_aligned.copy())
            data_stories[topic_id].append(group.loc[row_id]["story"])
            group.drop(row_id, inplace=True)
            topic_indices -= total_aligned 

    max_length = max(len(v) for v in all_coverage_data.values())
    for topic_id, coverage in all_coverage_data.items():
        while len(coverage) < max_length:
            all_coverage_data[topic_id].append(coverage[-1])


    df_coverage = pd.DataFrame(all_coverage_data).T 
    df_coverage = df_coverage.merge(topics[["all_oie", "alignment"]], how="left", left_index=True, right_index=True)
    df_coverage.to_json(os.path.join(args.data_dir, "source_coverage.jsonl"), orient="records", lines=True)
