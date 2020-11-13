import sys


K_FOLD = 6
USE_RULE = True


import pandas as pd
import os
import shutil
import json
from seqeval.metrics.sequence_labeling import get_entities
import zipfile

# os.chdir(os.getcwd())
sys.path.append(os.getcwd())

if USE_RULE:
    from ruler.regex_extractor import qq_extractor, vx_extractor

class PostProcess:
    def __init__(self, predict_file, txt_file_dir):
        
        predict_datas = []
        predict_data = []
        predict_example = []

        with open(predict_file, "r", encoding="utf8") as r:
            for line in r:
                if line.strip() == "":
                    labels = [e[-1] for e in predict_example]
                    predict_example = [[e[0], e[1], e[2], l] for e, l in zip(predict_example, labels)]
                    predict_data.extend(predict_example) 
                    predict_example = []
                    continue    
                else:
                    splits = line.strip().split(" ")
                    if len(splits) == 7:
                        token, file_id, poses, _, label, new_token, _ = splits
                        if new_token == "[UNK]":
                            # print(token)
                            pass
                        else:
                            assert token.lower() == new_token, [token, new_token]
                    else:
                        token = " "
                        file_id, poses, label, new_token, _ = splits
                        assert new_token == "[unused1]", [new_token]

                    if predict_data and file_id != predict_data[0][1]:
                        predict_datas.append(predict_data)
                        predict_data = []  
                    predict_example.append([token, file_id, poses, label])
            
        if predict_example:
            predict_data.extend(predict_example) 
        
        if predict_data:
            predict_datas.append(predict_data)
        self.predict_datas = predict_datas
        self.txt_file_dir = txt_file_dir

    
    def conll2brat(self): 
        submit_dict = {}
        for predict_data in self.predict_datas:
            file_id = predict_data[0][1]
            with open(os.path.join(self.txt_file_dir, f"{file_id}.txt"), "r", encoding="utf8") as r:
                text = r.read()

            labels = [d[-1] for d in predict_data]
            entities = get_entities(labels)

            for i in range(len(entities)):
                entities[i] = list(entities[i])
                start, end = entities[i][1], entities[i][2]
                entities[i][1] = int(predict_data[start][2])
                entities[i][2] = int(predict_data[end][2])
                entities[i] = tuple(entities[i])
            submit_dict[file_id] = entities
        return submit_dict


def convert_to_format(file_id, entities, txt_file_dir):
    with open(os.path.join(txt_file_dir, f"{file_id}.txt"), "r", encoding="utf8") as r:
        text = r.read()
    formats = []
    
    # 删掉模型预测的结果，加入规则预测的结果
    if USE_RULE:
        entities = [e for e in entities if e[0] not in ["vx", "QQ"]]
        # [start, end, tag]
        vx_entities = vx_extractor.extract(text)
        entities.extend([(tag, start, end-1) for start, end, tag in vx_entities])
        qq_entities = qq_extractor.extract(text)
        entities.extend([(tag, start, end-1) for start, end, tag in qq_entities])

    entities = sorted(entities, key=lambda x: (x[1], x[2]))


    for i, (label, real_start, real_end) in enumerate(entities):
        format_one = {
            "ID": file_id,
            "Category": label,
            "Pos_b": real_start,
            "Pos_e": real_end,
            "Privacy": text[real_start:real_end+1]
        }

        formats.append(format_one)
    return formats
    
        


def vote(submit_k_fold, n=None):
    '''投票'''
    import numpy as np
    if n is None:
        # 过半数票
        # n = np.ceil(len(submit_k_fold) / 2)
        n = len(submit_k_fold) // 2 + 1
        print(f"票数过半: {n}")

    submit = []
    all_file_id = set()
    for k_fold_examples in submit_k_fold:
        all_file_id.update(k_fold_examples.keys())

    all_file_id = sorted(list(all_file_id))

    for file_id in all_file_id:
        entity_count_dict = {}
        for i, one_fold_examples in enumerate(submit_k_fold):
            one_file_entities = one_fold_examples.get(file_id, [])
            for entity in one_file_entities:
                entity_count_dict[entity] = entity_count_dict.get(entity, 0) + 1
        entities = [e for e, c in entity_count_dict.items() if c >= n]
        submit.append((file_id, entities))
    return submit

def main():
    submit_k_fold = []
    txt_file_dir = "../user_data/data/test"
    for i in range(K_FOLD):
        # predict_file = f"../user_data/output_idcnn_crf/{i}_fold/test_predictions.txt"
        # submit_one_fold = PostProcess(predict_file, txt_file_dir).conll2brat()
        # submit_k_fold.append(submit_one_fold)
        
        predict_file = f"../user_data/output/output_layer_lstm_crf/{i}_fold/test_predictions.txt"
        if os.path.exists(predict_file):   
            submit_one_fold = PostProcess(predict_file, txt_file_dir).conll2brat()
            submit_k_fold.append(submit_one_fold)
        
        predict_file = f"../user_data/output/output_layer_idcnn_crf/{i}_fold/test_predictions.txt" 
        if os.path.exists(predict_file):    
            submit_one_fold = PostProcess(predict_file, txt_file_dir).conll2brat()
            submit_k_fold.append(submit_one_fold)

        # predict_file = f"../user_data/output/output_layer_output_crf/{i}_fold/test_predictions.txt"     
        # submit_one_fold = PostProcess(predict_file, txt_file_dir).conll2brat()
        # submit_k_fold.append(submit_one_fold)

    submit = vote(submit_k_fold)


    results = []
    for file_id, entities in submit:
        result = convert_to_format(file_id, entities, txt_file_dir)
        results.extend(result)
    
    to_csv = pd.DataFrame(results)

    to_csv.to_csv("../user_data/result/predict.csv", index=None)




    

if __name__ == "__main__":
    main()


            
                








