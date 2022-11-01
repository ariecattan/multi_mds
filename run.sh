# setup the path to raw_data and proces data
export raw_data_dir=raw_data
export process_dir=process_data

export data_name=$1
export split=$2

export raw_data=$raw_data_dir/$data_name/$split.jsonl
export data_dir=$process_dir/$data_name/$split

echo $raw_data
echo $data_dir

mkdir -p $data_dir
echo "processing" > $data_dir/processing.txt

echo "==> run openIE"
export openie_dir=$data_dir/oie
export openie_df=$data_dir/df_oie.csv

if [ -d "$openie_dir" ]; then
    echo "OpenIE already exist, skipping this step"
elif test -f "$openie_df"; then 
    echo "OpenIE already exist, skipping this step"
else 
    echo "Running openIE on both summaries and source documents"
    python extract_open_ie.py --raw_data $raw_data --data_dir $data_dir --gpu 0
fi


echo "==> prepare pairs"
export pairs_file=$data_dir/df_oie.csv
if test -f "$pairs_file"; then
    echo "Pairs already exist, skipping this step"
else 
    python prepare_oie_pairs.py --data_dir $data_dir
fi

echo "==> run SuperPAL"
export results=$data_dir/result_npy
if [ -d "$results" ]; then
    echo "SuperPAL scores already exist, skipping this step"
else 
    python get_superpal_scores.py --data_dir $data_dir --batch_size 16
fi

echo "==> build greedy subsets "
export subset_file=$data_dir/source_coverage.jsonl
if test -f "$subset_file"; then
    echo "Subsets already exist, skipping this step"
else 
    python build_greedy_subsets.py --data_dir $data_dir
fi


python get_aac_scores.py --data_dir $data_dir