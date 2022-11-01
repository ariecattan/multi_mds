# How "Multi" is Multi-Document Summarization? 

This repo contains the code for the paper: [How multi is Multi-Document Summarization (EMNLP 2022)](https://arxiv.org/pdf/2210.12688.pdf).


## Setup 

```bash
conda create --name multi_mds python=3.8 
conda activate multi_mds 
pip install -r requirements.txt 
```

If the setup fails on jsonnet, see [this](https://github.com/allenai/allennlp/issues/1969) issue.  


## Preprocessing data 

You should pre-preprocess your dataset into `jsonl` format where each lines includes the following fields:
* `document`: a List of source documents 
* `summary`: a list of reference summaries
* `topic_id`: instance id 


## Compute the AAC score and curves

There are several steps for computing the AAC score:
1. extract the openIE from all source documents and the summary
2. prepare pairs of OpenIE
3. compute alignment scores between source and summary propositions for each topic
4. build greedily the maximally covering subsets of source documents
5. compute the Area Above the Curve and save the coverage plot. 


You can run a single command that will compute all steps together, while skipping accomplished steps (edit the path of `raw_data_dir` and `process_dir`): 

```bash
bash run.sh [preprocessed_data] [dir_path] 
```

Alternatively, you can run each step separately, as follows: 

1. Extract all Open IE tuples from the summary and the source documents. 

```bash
export raw_data= # path to jsonl file 
export data_dir= # output dir

python extract_open_ie.py --raw_data $raw_data \
                          --data_dir $data_dir \
                          --gpu 0 
```

This script will create a directory `$data_dir/oie` with the propositions from the summary and the documents. 

2. Prepare pairs:

```bash 
python prepare_oie_pairs.py --data_dir $data_dir
```

This script will create a file `$data_dir/pairs.pickle` with all possible pairs of open IE.

3. Compute alignment scores between source and summary propositions for each topic:

```bash
python get_superpal_scores.py --data_dir $data_dir \
                              --model biu-nlp/superpal \
                              --device_ids 0,1,2,3 \
                              --batch_size 64
```

This script will run the alignment model on the `$data_dir/pairs.pickle` and save the results in the directory `$data_dir/result_npy`. 

4. Build greedy subsets of documents that maximize coverage 

```bash
python build_greedy_subsets.py --data_dir $data_dir 
```

5. Compute AAC score and save plot in `$data_dir/plot.png`. 

```bash
python get_aac_scores.py --data_dir $data_dir
```



## Citation 

```bash
@inproceedings{Wolhandler2022HowI,
  title={How "Multi" is Multi-Document Summarization?},
  author={Ruben Wolhandler and Arie Cattan and Ori Ernst and Ido Dagan},
  booktitle={EMNLP},
  year={2022}
}
```