from allennlp.predictors.predictor import Predictor
from datasets import load_dataset
import argparse
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from tqdm import tqdm
import os
import jsonlines
import json
import glob
from utils import *
from rich.progress import track, Progress




def write_oie(args, splitter, out_dir, predictor, tgt_text='summary', column_text='document'):
    ########### create output directory if doesn't exist ##############################
    if not os.path.isdir(os.path.join(out_dir, column_text)):
        os.makedirs(os.path.join(out_dir, column_text))
    if not os.path.isdir(os.path.join(out_dir, tgt_text)):
        os.makedirs(os.path.join(out_dir, tgt_text))
    ###################################################################################

    with jsonlines.open(args.raw_data) as f:
        dataset = [x for x in f]

    
    
    for topic_json in track(dataset):
        topic_id = int(topic_json["topic_id"])
        summary = topic_json["summary"]

        if os.path.exists(os.path.join(out_dir, tgt_text, f"{topic_id}.jsonl")):
            continue

        sentences_summary = splitter.split_sentences(summary)

        # proess summary oie
        f_tgt = jsonlines.open(os.path.join(out_dir, tgt_text, f"{topic_id}.jsonl"), 'w')
        json_input_tgt = [{"sentence": sentences_summary[k]} for k in range(len(sentences_summary))]
        result_list = predictor.predict_batch_json(json_input_tgt)
        assert (len(result_list) == len(json_input_tgt))  # we return for each sentence list of oie

        # Keep idx start of each sentences of the summary
        idx_start_summ = 0
        for k, result in enumerate(result_list):
            if sentences_summary[k] == '':  # bug of sentences split
                result['sentences'] = sentences_summary[k]
                result['scuSentCharIdx'] = idx_start_summ
                continue

            while summary[idx_start_summ] == ' ' or summary[idx_start_summ] == '\n':
                idx_start_summ += 1
            result['sentences'] = sentences_summary[k]
            result['scuSentCharIdx'] = idx_start_summ

            assert (summary[idx_start_summ] == result['sentences'][0])  # check if the start index is the good one
            idx_start_summ = idx_start_summ + len(sentences_summary[k])

        f_tgt.write(result_list)


        if args.only_summary:  # for system summaries, we predict openIE only on the summaries
            continue

        doc_files = topic_json["documents"]
        f_src = jsonlines.open(os.path.join(out_dir, column_text, f"{topic_id}.jsonl"), 'w')
        for story_id, document in enumerate(doc_files):
            document = document.strip().replace("\n\n", " ").replace("\n", " ")
            sentences_document = splitter.split_sentences(document)

            json_input_src = [{"sentence": sentences_document[k]} for k in range(len(sentences_document))]
            result_list = predictor.predict_batch_json(json_input_src)
            assert (len(result_list) == len(json_input_src))  # we return for each sentence list of oie

            # Keep idx start of each sentences of the summary
            idx_start = 0
            for k, result in enumerate(result_list):
                if sentences_document[k] == '':  # bug of sentences split
                    result['sentences'] = sentences_document[k]
                    result['scuSentCharIdx'] = idx_start
                    continue

                while document[idx_start] == ' ' or document[idx_start] == '\t':
                    idx_start += 1
                result['sentences'] = sentences_document[k]
                result['scuSentCharIdx'] = idx_start

                assert (document[idx_start] == result['sentences'][0])  # check if the start index is the good one
                idx_start = idx_start + len(sentences_document[k])

            f_src.write(result_list)
        f_src.close()
        f_tgt.close


def main(args):
    path_oie_model = "https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz"
    predictor = Predictor.from_path(path_oie_model, cuda_device=args.gpu)
    splitter = SpacySentenceSplitter()
    splitter.spacy.max_length *= 100
    '''
    DATASET STRUCTURE: jsonlines file that each line correspond to one topic
    {"topic_id":"topic_id", "documents": [doc1, doc2, doc3....], "document_name" : [name_doc1, name_doc2, name_doc3 ....]}
    .
    .
    .
    '''

    out_dir = os.path.join(args.data_dir, 'oie')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    write_oie(args, splitter, out_dir, predictor)
#python oie_main.py --set_oie validation --gpus 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", type=str, help="path to jsonl file of raw data")
    parser.add_argument("--data_dir", type=str, help="path to directory to save Open IE")
    parser.add_argument("--only_summary", action="store_true", default=False)
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()

    main(args)