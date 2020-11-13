# html删掉
# 逗号 引号 冒号 分号 括号中文转英文
# 尚不明确删掉
# (), [], <>里面全英文的


import json
import os
import re
import sys
from glob import glob

import pandas as pd
from seqeval.metrics.sequence_labeling import get_entities

sys.path = [os.getcwd()] + sys.path


USE_EXTERNAL = True
external_dir_path = "../user_data/data/external_data"

user_data_dir = "../user_data/data"
test_dir_path = "../user_data/data/test"
train_dir_path = "../user_data/data/train/"


try:
    K_FOLD = int(sys.argv[1])
    MAX_LEN = int(sys.argv[2])
except Exception as e:
    print(e)
    K_FOLD = 10
    MAX_LEN = 512

print(f"K_FOLD: {K_FOLD}, MAX_LEN: {MAX_LEN}")

LABELS_LIST = [
        "position",        
        "name",       
        "organization",
        "company",   
        "address",        
        "movie",        
        "game",          
        "government",
        "scene",     
        "book",           
        "mobile",
        "email",          
        "QQ",
        "vx"
]


BLANK_RE = re.compile("\s")

ZH_RE = re.compile("[\u4e00-\u9fa5]")

# REPLACE_MAP中的key将会被替换为value
REPLACE_MAP = {}

# 如果SPECIAL_RE不为None，在SPECIAL_RE中且不在KEEP_LIST中的将会被删除
# SPECIAL_RE = re.compile("[^\u4e00-\u9fa5_a-z_A-Z_0-9]")
SPECIAL_RE = None
KEEP_LIST = set("!#$%&\()*+,-./:;<=>?@[\\]^_`{|}~！#￥%&？《》{}“”，：‘’。（）·、；【】")

# pre_segment_process的操作顺序 SPECIAL_RE
# post_segment_process的操作顺序 REFINE_RE -> SPECIAL_RE -> REMOVE_RE
REFINE_RE = []

# 删除下列字符，支持正则表达式
REMOVE_RE = []



class PreProcessBase:
    def __init__(self, txt_file, ann_file=None):
        # read file
        file_id = os.path.basename(txt_file).split(".")[0]
        with open(txt_file, "r", encoding="utf8") as r:
            text = r.read()
        if ann_file:
            ann = pd.read_csv(ann_file)
            ann["id_"] = ann.ID
            ann["label"] = ann.Category
            
            def get_span(row):
                Pos_b = row.Pos_b
                Pos_e = row.Pos_e
                if Pos_b < 0:
                    Pos_b = 0
                if Pos_e >= len(text):
                    Pos_e = len(text) - 1
                return [Pos_b, Pos_e+1]

            ann["span"] = ann.apply(get_span, axis=1)
            ann["mention"] = ann.span.apply(lambda x: text[x[0]:x[1]])
            ann = ann.drop(["Pos_b", "Pos_e", "Category", "ID", "Privacy"], axis=1)
            
        else:
            ann = None

        # 所有空格转英文空格
        text = BLANK_RE.sub(" ", text)
        # 替换特殊字符
        for k, v in REPLACE_MAP.items():
            text = text.replace(k, v)

        text = text.lower()
        self.text = text
        self.ann = ann
        self.file_id = file_id

    def _brat2conll(self, text, ann):
        conll_text = []
        conll_label = []
        if ann is not None:
            for _, row in ann.iterrows():
                label = row.label
                assert label in LABELS_LIST, [label]
                start, end = row.span

                text = list(text)
                try:
                    text[start] = f"<{label}>" + text[start]
                    text[end-1] = text[end-1] + f"</{label}>"
                except Exception as e:
                    print(text)
                    print(start)
                    print(end)
                    raise e

        is_inner = False
        cur_label = None
        # brat转为conll格式
        for word in text:
            match_start = re.search(f"^<({'|'.join(LABELS_LIST)})>", word)
            match_end = re.search(f"</({'|'.join(LABELS_LIST)})>$", word)
            if match_start:
                label = match_start.group(1)
                assert label in LABELS_LIST, [label]
                conll_label.append(f"B-{label}")
                if not match_end:
                    is_inner = True
                cur_label = label
            elif is_inner:
                conll_label.append(f"I-{cur_label}")
                if match_end:
                    is_inner = False
                    cur_label = None
            else:
                conll_label.append("O")

            word = re.sub(f"</?({'|'.join(LABELS_LIST)})>", "", word)
            conll_text.append(word)
        return conll_text, conll_label

    def convert_char_to_special_token(self, sentences, char, token):
        '''将token转为特殊token'''
        for i in range(len(sentences)):
            for j in range(len(sentences[i])):
                if sentences[i][j][0] == char:
                    sentences[i][j][0] = token
        return sentences

    def segment_sentences(self, conll_text, poses, conll_label, file_id, ann=True):
        '''分句'''
        sentences = []
        sentence = []
        for i, (word, pos, label) in enumerate(zip(conll_text, poses, conll_label)):
     
            if word in "。;":
                # 标点分句
                if ann is not None:
                    sentence += [[word, file_id, pos, label]]
                else:
                    sentence += [[word, file_id, pos]]
                sentences.append(sentence)
                sentence = []
            else:
                if ann is not None:
                    sentence += [[word, file_id, pos, label]]
                else:
                    sentence += [[word, file_id, pos]]
        if sentence:
            sentences.append(sentence)

        if MAX_LEN < 0:
            sentences = self.strip(sentences, " ")
            sentences = self.filter_zh_sentences(sentences)
        return sentences

    @staticmethod
    def strip(string_list, regexp):
        rets = []
        for string in string_list:
            while string and re.search(regexp, string[0][0]):
                string.pop(0)
            while string and re.search(regexp, string[-1][0]):
                string.pop()
            rets.append(string)
        return rets

    def filter_zh_sentences(self, sentences):
        '''没出现中文的句子删掉'''
        ret = []
        # 一个中文都没有的句子删掉
        for sentence in sentences:
            if ZH_RE.search("".join(c[0] for c in sentence)):
                # if ZH_RE.search("".join(c[0] for c in sentence)):
                ret.append(sentence)
        return ret

    def assert_pos(self, conll):
        '''检查和brat格式数据位置是否对齐'''
        text = self.text
        for terms in conll:
            for term in terms:
                word, pos = term[0], term[2]
                if word != "[unused1]":
                    assert word == text[pos], [term, text[pos]]

    def get_remove_index(self, regexps, string):
        '''根据正则删除句子里的一些token'''
        remove_index_list = set()
        string_copy = string[:]
        for regexp in regexps:
            finditer = re.finditer(regexp, string)
            for search in finditer:
                start, end = search.span()
                remove_index_list.update(list(range(start, end)))
            string_copy = re.sub(regexp, "", string_copy)

        remove_index_list = list(remove_index_list)
        remove_index_list.sort()
        return remove_index_list

    def combine_sentences(self, sentences):
        rets = []
        ret = []
        for sentence in sentences:
            if len(ret) + len(sentence) <= MAX_LEN:
                ret += sentence
            else:
                rets.append(ret)
                ret = sentence
        if ret:
            rets.append(ret)
        return rets

    def remove_special_token(self, string):
        remove_index_list = []
        if SPECIAL_RE:
            for index, c in enumerate(string):
                if SPECIAL_RE.match(c) and (c not in KEEP_LIST):
                    remove_index_list.append(index)
        return remove_index_list


class PreProcess(PreProcessBase):
    def brat2conll(self):
        '''brat格式转conll'''
        text = self.text
        ann = self.ann
        file_id = self.file_id
        info = {}

        conll_text = []
        conll_label = []
        info = {}

        # brat转conll格式
        conll_text, conll_label = self._brat2conll(text, ann)

        assert len(conll_text) == len(conll_label)

        # 预处理
        sentences = list(zip(conll_text, list(
            range(len(conll_text))), conll_label))
        sentences = self.pre_segment_preprocess(sentences)
        if not sentences:
            return sentences, {"del_sen": 0}
        conll_text, poses, conll_label = zip(*sentences)

        # 分句
        sentences = self.segment_sentences(
            conll_text, poses, conll_label, file_id, ann is not None)
        if not sentences:
            return sentences, {"del_sen": 0}
        
        length_segment = len(sentences)
        # 后处理
        sentences = self.post_segment_preprocess(sentences)
        if not sentences:
            return sentences, {"del_sen": 0}
        length_post = len(sentences)
        info["del_sen"] = length_segment - length_post

        if MAX_LEN > 0:
            sentences = self.combine_sentences(sentences)

            sentences = self.strip(sentences, " ")
            sentences = self.filter_zh_sentences(sentences)

        sentences = self.convert_char_to_special_token(
            sentences, " ", "[unused1]")

        # 检查位置是否对应
        self.assert_pos(sentences)

        return sentences, info

    def pre_segment_preprocess(self, sentence):
        '''分句前的处理'''

        string = "".join([c[0] for c in sentence])
        remove_index_list = []

        # 先是一些定制化的操作
        # 删除括号里是英文的和html等
        # remove_index_list = self.get_remove_index(
        #     ["[<\[(][^\u4e00-\u9fa5]*?[>\])]"], string)
        
        # 删除网址
        # remove_index_list.extend(self.get_remove_index(['(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]'], string))
        # remove_index_list.extend(self.get_remove_index(["www.[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]"], string))
        
        # 删除无关字符
        remove_index_list.extend(self.remove_special_token(string))

        remove_index_list = list(remove_index_list)
        remove_index_list.sort()

        if remove_index_list:
            sentence = self.remove_from_index(sentence, string, remove_index_list)

        return sentence

    def remove_from_index(self, sentence, string, remove_index_list):
        tmp = []
        bad_index_start = None # 如果把包含实体的部分删除了需要打印出来
        for index, term in enumerate(sentence):
            if index not in remove_index_list:
                if bad_index_start is not None:
                    print(string)
                    print([s for s in sentence[bad_index_start:index]])
                    print(" ")

                    bad_index_start = None
                tmp.append(term)
            else:
                if bad_index_start is not None:
                    continue
                elif sentence[index][-1] != "O":
                    bad_index_start = index
        if bad_index_start is not None:
            print(string)
            print([s[-1] for s in sentence[bad_index_start:]])
            print(" ")

        sentence = tmp
        return sentence

    def post_segment_preprocess(self, sentences):
        '''分句后的处理'''
        # refine文本
        for i, sentence in enumerate(sentences):
            string = "".join([c[0] for c in sentence])
            remove_index_list = self.get_remove_index(REFINE_RE, string)

            remove_index_list.extend(self.remove_special_token(string))

            remove_index_list = list(remove_index_list)
            remove_index_list.sort()

            if remove_index_list:
                sentence = self.remove_from_index(
                    sentence, string, remove_index_list)

            sentences[i] = sentence

        # 删除一些无关句子
        for i, sentence in enumerate(sentences):
            string = "".join([c[0] for c in sentence])
            remove_index_list = self.get_remove_index(REMOVE_RE, string)

            remove_index_list = list(remove_index_list)
            remove_index_list.sort()

            if remove_index_list:
                sentence = self.remove_from_index(
                    sentence, string, remove_index_list)
            sentences[i] = sentence

        if MAX_LEN < 0:
            sentences = self.strip(sentences, " ")
            sentences = self.filter_zh_sentences(sentences)
        return sentences


def run(file_name_list, mode):
    import datetime
    from tqdm import tqdm
    print(f"------------------start For {mode}----------------------")
    conlls = []
    all_del_sen = 0
    all_label_num = 0
    for file_name in sorted(file_name_list):
        txt_name = file_name
        if mode != "test":
            ann_name = file_name[:-4] + ".csv"
            preprocess = PreProcess(txt_name, ann_name)
            all_label_num += preprocess.ann.shape[0]
        else:
            preprocess = PreProcess(txt_name)
        conll, info = preprocess.brat2conll()
        all_del_sen += info["del_sen"]
        conlls.extend(conll)

    max_len = max([len(s)+2 for s in conlls])

    label_num = 0
    for conll in conlls:
        label = [c[-1] for c in conll]
        label_num += len(get_entities(label))

    print(
        f'''
            句子总数   : {len(conlls)}
            删除句子个数: {all_del_sen}
            句子最大长度: {max_len}
            原始实体个数: {all_label_num}
            当前实体个数: {label_num}
        '''
    )
    # 打印lanbel的分布情况
    label_dict = {}
    for sentence in conlls:
        label = [s[-1] for s in sentence]
        for entity in get_entities(label):
            entity = entity[0]
            label_dict[entity] = label_dict.get(entity, 0) + 1

    if mode != "test":
        print("实体分布情况：")
        totle_num = 0
        for l in LABELS_LIST:
            if l not in label_dict:
                label_dict.update({l: 0})
        totle_num = sum(label_dict.values())
        for k, v in sorted(label_dict.items(), key=lambda x: x[0]):
            print(f"{k}:\t{v}\t{v/totle_num}")
        print(f"All: {totle_num}")

    print("-------------------END----------------------\n")
    return conlls


def save_to_file(conlls, path, mode):
    if not os.path.exists(path):
        os.mkdir(path)

    string = ""
    for conll in conlls:
        for term in conll:
            term = [str(t) for t in term]
            string += " ".join(term) + "\n"
        string += "\n"

    output_file = os.path.join(path, f"{mode}.txt")
    with open(output_file, "w", encoding="utf8") as w:
        w.write(string)
    print(f"save to {output_file}")
    return conlls


def main_k_fold(mode):
    import numpy as np
    from sklearn.model_selection import KFold


    k_fold_dir = os.path.join(user_data_dir, f"k_fold_{K_FOLD}_{MAX_LEN}")
    if not os.path.exists(k_fold_dir):
        os.mkdir(k_fold_dir)

    

    k_fold = KFold(n_splits=K_FOLD, shuffle=True, random_state=2333)

    if "predict" in mode:
        test_file_list = glob(os.path.join(test_dir_path, "*.txt"))
        conlls_test = run(test_file_list, "test")
        save_to_file(conlls_test, k_fold_dir, "test")

    train_file_list = glob(os.path.join(train_dir_path, "*.txt"))
    train_file_array = np.array(train_file_list)

    external_file_list = []
    if USE_EXTERNAL:
        external_file_list = glob(os.path.join(external_dir_path, "*.txt"))
    

    if "train" in mode:
        for i, (train_list, dev_list) in enumerate(k_fold.split(train_file_list)):
            print(f"for {i}-fold")
            output_dir = os.path.join(k_fold_dir, f"fold_{i}")   
            train_list = train_file_array[train_list].tolist() + external_file_list
            dev_list = train_file_array[dev_list].tolist()
            conlls_train = run(train_list, "train")
            conlls_dev = run(dev_list, "dev")

            save_to_file(conlls_train, output_dir, "train")
            save_to_file(conlls_dev, output_dir, "dev")   
             

if __name__ == "__main__":
    mode = sys.argv[3]
    main_k_fold(mode)

