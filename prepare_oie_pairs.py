import os
import pickle
from tqdm import tqdm
import jsonlines
import pandas as pd
import argparse
from utils import *

class OpenIE:
    def __init__(
        self,
         oie,
        tokenized_sentence,
        original_sentence,
        oie_txt,
        oie_offset,
        topic_id,
        story_id,
        is_summary
     ):
        self.is_summary: bool = is_summary
        self.story_id: int = story_id
        self.topic_id: int = topic_id
        self.tokenized_sentence: str = tokenized_sentence
        self.oie = oie
        self.arg0 = True if 'B-ARG0' in oie['tags'] else False
        self.arg1 = True if 'B-ARG1' in oie['tags'] else False
        self.arg2 = True if 'B-ARG2' in oie['tags'] else False
        self.oie_txt : str = oie_txt
        self.oie_offset = oie_offset
        self.original_sentence : str = original_sentence

    def oie2str(self):
        sent = ''
        for key in self.clean_oie.keys():
            sent += self.clean_oie[key]['span'] + ' '
        return sent[:-1] if len(sent) != 0 else sent

    def to_dict(self):
        return {
            'oie_txt': self.oie_txt,
            'is_summary': self.is_summary,
            'story_id': self.story_id,
            'topic_id': self.topic_id,
            'tokenized_sentence': self.tokenized_sentence,
            'original_sentence': self.original_sentence,
            'oie': self.oie,
            'arg0': self.arg0,
            'arg1': self.arg1,
            'arg2': self.arg2,
            'oie_offset': self.oie_offset
        }


def generate_scu_oie_multiSent(oies, topic_id, story_id=None, is_summary=True):
    """ Given a scu sentence retrieve SCUs (OIEs)
    
    The input should be a list of dictionaries with the following fields:
        'scuSentence' #sentence text
        'scuSentCharIdx' # character offset of the beginning of the sentence w.r.t the beginning of the document 
        'scuText' # The OIE text would be written here. 
        'scuOffsets' # The character offset of the OIE w.r.t the beginning of the document would be written here

        'oies' list of oies from one document (summary or one of the document input)
        'story_id' # None if the doc is summary else the story id of the doc
        'topic_id' #the topic id of the document 
   
    """
    
    KEY_sent_char_idx = 'scuSentCharIdx'

    scu_list = []
    for oie in oies:
        sentence = oie['sentences']
        sentence = sentence.replace(u'\u00a0', ' ')
        # ipdb.set_trace()
        if not oie:  # if list is empty
            continue

        # if  sentence =='Johnson\'s new TV show, ``The Magic Hour,\'\' is just one aspect of a busy life:  -- HIS HEALTH: While by no means cured, he owes the appearance of remarkable health to a Spartan lifestyle and modern medicine.':
        #     print('here')
        scus = oie['verbs']
        in_sentence_scu_dict = {}
        tokens = oie['words']
        for scu in scus:
            tags = scu['tags']
            words = []
            if not ("B-ARG1" in tags or "B-ARG2" in tags or "B-ARG0" in tags):
                continue
            sub_scu_offsets = []
            scu_start_offset = None
            offset = 0
            initialSpace = 0
            spaceAfterToken = 0
            if '\t' in sentence:
                sentence = sentence.replace('\t',' ')
            while sentence[offset + initialSpace] == ' ' or sentence[offset + initialSpace] == '\n' or sentence[offset + spaceAfterToken] == '\t': #check
                initialSpace += 1  ## add space if exists, so 'offset' would start from next token and not from space
            offset += initialSpace
            for ind, tag in enumerate(tags):
                # if "ARG0" in tag or "ARG1" in tag or "V" in tag:
                assert (sentence[offset] == tokens[ind][0])
                if "O" not in tag:
                    if scu_start_offset is None:
                        scu_start_offset = oie[KEY_sent_char_idx] + offset

                        assert(sentence[offset] == tokens[ind][0])

                    words.append(tokens[ind])
                else:
                    if scu_start_offset is not None:
                        spaceBeforeToken = 0
                        while sentence[offset-1-spaceBeforeToken] == ' ' or sentence[offset-1-spaceBeforeToken] == '\n' or sentence[offset-1-spaceBeforeToken] == '\n\n':
                            spaceBeforeToken += 1## add space if exists
                        if sentence[offset] == '.' or sentence[offset] == '?':
                            dotAfter = 1 + spaceAfterToken
                            dotTest = 1
                        else:
                            dotAfter = 0
                            dotTest = 0
                        scu_end_offset = oie[KEY_sent_char_idx] + offset - spaceBeforeToken + dotAfter

                        if dotTest:
                            assert (sentence[offset - spaceBeforeToken + dotAfter -1] == tokens[ind-1+ dotTest][0]) #check only the dot, the start of the token
                        else:
                            assert (sentence[offset - spaceBeforeToken + dotAfter - 1] == tokens[ind - 1 + dotTest][-1])  #check end of token
                        sub_scu_offsets.append([scu_start_offset, scu_end_offset])
                        scu_start_offset = None


                ## update offset

                offset += len(tokens[ind])
                if ind < len(tags) - 1: #if not last token
                    spaceAfterToken = 0
                    while sentence[offset + spaceAfterToken] == ' ' or sentence[offset + spaceAfterToken] == '\n' or sentence[offset + spaceAfterToken] == '\t':
                        spaceAfterToken += 1## add space after token if exists, so 'offset' would start from next token and not from space
                    offset += spaceAfterToken

            if scu_start_offset is not None: #end of sentence
                scu_end_offset = oie[KEY_sent_char_idx] + offset
                sub_scu_offsets.append([scu_start_offset, scu_end_offset])
                scu_start_offset = None



            # if len(words) <= 3:
            #     continue
            #scuText = "...".join([sentence[strt_end_indx[0] - oie[KEY_sent_char_idx]:strt_end_indx[1] - oie[KEY_sent_char_idx]] for strt_end_indx in sub_scu_offsets])
            scuText = " ".join([sentence[strt_end_indx[0] - oie[KEY_sent_char_idx]:strt_end_indx[1] - oie[KEY_sent_char_idx]] for strt_end_indx in sub_scu_offsets])
            #assert(scuText==" ".join([sentence[strt_end_indx[0]:strt_end_indx[1]] for strt_end_indx in sub_scu_offsets]))
            in_sentence_scu_dict[scuText] = sub_scu_offsets

        notContainedDict = checkContained(in_sentence_scu_dict, sentence, oie[KEY_sent_char_idx])


        for scuText, binaryNotContained in notContainedDict.items():
            scu_offsets = in_sentence_scu_dict[scuText]
            if binaryNotContained:
                tmp = OpenIE(scu, tokens, sentence, scuText, scu_offsets, topic_id, story_id, is_summary)
                scu_list.append(tmp.to_dict())
  
    return scu_list

def list_oie_from_path(oie_doc_path, is_summary):
    list_oie = []
    topic_id = int(oie_doc_path.split('.')[0].split('/')[-1])
    story_id = 0
    with jsonlines.open(oie_doc_path) as reader:
        docs = list(reader)
    for doc_oies in docs:
        scus = generate_scu_oie_multiSent(doc_oies, topic_id, story_id, is_summary=is_summary)
        list_oie.extend(scus)
        story_id += 1
    return list_oie



def create_df_oie(dir_sums, dir_docs):
    oie_list = []
    sums_paths = os.listdir(dir_sums)
    docs_paths = os.listdir(dir_docs)
    for sum_path in tqdm(sums_paths):
        oie_list.extend(list_oie_from_path(os.path.join(dir_sums,sum_path), True))
    for doc_path in tqdm(docs_paths):
        oie_list.extend(list_oie_from_path(os.path.join(dir_docs,doc_path), False))
    df = pd.DataFrame.from_records(oie_list)
    return df

def create_pairs(openie_df, data_dir):
    f = open(os.path.join(data_dir, "pairs.pickle"), 'wb')
    number_of_pairs = 0
    for topic_id in tqdm(range(max(openie_df['topic_id'])+1)):
        summaries = [(index, row) for index, row in openie_df[openie_df['topic_id']==topic_id].iterrows() if row['is_summary'] == True]
        documents = [(index, row) for index, row in openie_df[openie_df['topic_id']==topic_id].iterrows() if row['is_summary'] == False]
        for sum_ in summaries:
            for doc in documents:
                pairs = {
                    # 'pairs': sum[1]['oie_txt'] + ' </s><s> ' + doc[1]['oie_txt'],
                    'pairs': doc[1]['oie_txt'] + ' </s><s> ' + sum_[1]['oie_txt'],
                    'index_sum': sum_[0],
                    'index_doc': doc[0]
                }
                pickle.dump(pairs, f)
                number_of_pairs+=1
    print(f"number of pairs: {number_of_pairs}")
    f.close()
    
    with open(os.path.join(data_dir, "num_of_pairs.txt"), "w") as f:
        f.write(str(number_of_pairs))
        

def main(args):
    dir_docs = os.path.join(args.data_dir, "oie", "document")
    dir_sums = os.path.join(args.data_dir, "oie", "summary")
    df = create_df_oie(dir_sums, dir_docs)
    df.to_csv(os.path.join(args.data_dir, "df_oie.csv"))
    create_pairs(df, args.data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    args = parser.parse_args()
    main(args)



