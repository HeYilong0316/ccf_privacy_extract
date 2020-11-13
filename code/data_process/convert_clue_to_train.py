import json
import pandas as pd
import os


save_dir = "../user_data/data/external_data/"

file_id = 5000

def convert_to_train(data_path):
    global file_id
    with open(data_path, "r", encoding="utf8") as r:
        for line in r:
            line = line.rstrip("\n")
            data = json.loads(line)

            text = data["text"]
            label = data["label"]

            ann_df = []
            for tag, entity in label.items():
                for mention, spans in entity.items():
                    for span in spans:
                        ann_df.append({
                            "ID": file_id,
                            "Category": tag,
                            "Pos_b": span[0],
                            "Pos_e": span[1],
                            "Privacy": mention
                        })
            with open(os.path.join(save_dir, f"{file_id}.txt"), "w", encoding="utf8") as w:
                w.write(text)
            pd.DataFrame(ann_df).to_csv(os.path.join(save_dir, f"{file_id}.csv"), index=None)
            file_id += 1

convert_to_train(data_path = "../data/cluener/train.json")
convert_to_train(data_path = "../data/cluener/dev.json")
  
print(file_id)

