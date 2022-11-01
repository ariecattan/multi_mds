import argparse
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
from torch.utils.data import IterableDataset
import pickle
import logging
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Any, Optional
import socket



class PairsOieDataset(IterableDataset):
    def __init__(self, args, tokenizer):
        self.picklename = os.path.join(args.data_dir, "pairs.pickle")
        self.tokenizer = tokenizer
        with open(os.path.join(args.data_dir, "num_of_pairs.txt"), "r") as f:
            self.num_of_pairs = int(f.readlines()[0])
        self.counter = 0

    def _iter_pickle(self):
        with open(self.picklename, "rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                    self.counter += 1
                except EOFError:
                    break

    def __iter__(self):
        return self._iter_pickle()


    def tokenize_batch(self, batch):
        pairs = list(map(lambda data: data['pairs'], batch))
        index_sum = list(map(lambda data: data['index_sum'], batch))
        index_doc = list(map(lambda data: data['index_doc'], batch))
        tokens = self.tokenizer(pairs, padding=True, truncation=True)
        input_ids = torch.tensor(tokens['input_ids'])
        attention_mask = torch.tensor(tokens['attention_mask'])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "index_sum": index_sum,
            "index_doc": index_doc
        }


class Aligner(pl.LightningModule):
    def __init__(self, args):
        super(Aligner, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.model = AutoModelForSequenceClassification.from_pretrained(args.model)
        self.df = pd.read_csv(os.path.join(args.data_dir, "df_oie.csv"))


    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None):
        scores = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        return {
            "scores": torch.softmax(scores.logits, dim=1)[:, 1],
            "index_sum": batch['index_sum'],
            "index_doc": batch['index_doc'],
            "topic_id": np.array(self.df.iloc[batch['index_sum']]['topic_id']),
            "story_id": np.array(self.df.iloc[batch['index_doc']]['story_id'])
        }



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--model", type=str, default="biu-nlp/superpal")
    parser.add_argument("--device_ids", type=str, default="0,1,2,3")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    root_logger = logging.getLogger()
    logger = root_logger.getChild(__name__)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt="%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    
    logger.info("init model and data..")
    aligner = Aligner(args)
    dataset = PairsOieDataset(args, aligner.tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.tokenize_batch, num_workers=1)
    device_ids = [int(x) for x in args.device_ids.split(',')]
    trainer = pl.Trainer(gpus=device_ids, accelerator='ddp')

    logger.info("predict scores")
    results = trainer.predict(aligner, dataloader)
    results_scores = torch.cat([x['scores'].clone() for x in results])
    results_index_doc = torch.cat([x['index_doc'].clone() for x in results])
    results_index_sum = torch.cat([x['index_sum'].clone() for x in results])
    results_topic_id = torch.cat([x['topic_id'].clone() for x in results])
    results_story_id = torch.cat([x['story_id'].clone() for x in results])

    np_dir = os.path.join(args.data_dir, "result_npy")
    if not os.path.exists(np_dir):
        os.makedirs(np_dir)

    logger.info("save scores")
    np.save(os.path.join(np_dir, '{}_scores.npy'.format(socket.gethostname())),
            results_scores.cpu().numpy(), allow_pickle=True)
    np.save(os.path.join(np_dir, '{}_index_doc.npy'.format(socket.gethostname())),
            results_index_doc.cpu().numpy(), allow_pickle=True)
    np.save(os.path.join(np_dir, '{}_index_sum.npy'.format(socket.gethostname())),
            results_index_sum.cpu().numpy(), allow_pickle=True)
    np.save(os.path.join(np_dir, '{}_story_ids.npy'.format(socket.gethostname())),
            results_story_id.cpu().numpy(), allow_pickle=True)
    np.save(os.path.join(np_dir, '{}_topic_ids.npy'.format(socket.gethostname())),
            results_topic_id.cpu().numpy(), allow_pickle=True)