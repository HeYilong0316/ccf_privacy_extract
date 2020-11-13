import re

class BaseExtractor(object):
    regex_list = [
        re.compile(r"base_regex")
    ]
    tag_name = "base"
    def extract(self, text):
        string = text.lower()
        ret = []
        for regex in self.regex_list:
            for find in regex.finditer(string):
                ret.append([find.start(1), find.end(1), self.tag_name])
        return ret


class VxExtractor(BaseExtractor):
    regex_list = [
        re.compile(r"微信(?:.{0,5}号)?[:：]?([a-z_A-z_0-9\_-]{6,20})")
    ]
    tag_name = "vx"
vx_extractor = VxExtractor()

class QQExtractor(BaseExtractor):
    regex_list = [
        re.compile(r"[Qq][Qq]?(?:.{0,5}群)?[:：]?([1-9][0-9]{5,11})"),
        # re.compile(r"微信号?[:：]?([1-9][0-9]{5,11})") # 微信号不能以数字开头，此时应为qq
    ]
    tag_name = "QQ"
qq_extractor = QQExtractor()


class MobileExtractor(BaseExtractor):
    regex_list = [
        # 大陆手机
        re.compile(r"((?:[(（]\+?86[)）])?1(?:3\d{3}|5[^4\D]\d{2}|8\d{3}|7(?:[0-35-9]\d{2}|4(?:0\d|1[0-2]|9\d))|9[0-35-9]\d{2}|6[2567]\d{2}|4(?:(?:10|4[01])\d{3}|[68]\d{4}|[579]\d{2}))\d{6})"),
        # 台湾手机
        re.compile(r"[(（]886[）)]?09\d{8}")
    ]
    tag_name = "mobile"
mobile_extractor = MobileExtractor()



if __name__ == "__main__":
    # text = "微信:1827967769"
    # ret = qq_extractor.extract(text)
    # for r in ret:
    #     print(ret, text[r[0]:r[1]])

    # text = "qq:2308713823"
    # ret = qq_extractor.extract(text)
    # for r in ret:
    #     print(ret, text[r[0]:r[1]])

    # text = "微信:hyl222"
    # ret = vx_extractor.extract(text)
    # for r in ret:
    #     print(ret, text[r[0]:r[1]])

    def test_extractor(extractor, tag):
        from glob import glob
        import os
        import pandas as pd

        # extractor = vx_extractor
        # tag = "vx"
        train_dir = "../user_data/data/train"
        pred_set = set()
        labe_set = set()
        for file_path in glob(os.path.join(train_dir, "*.txt")):
            file_id = os.path.basename(file_path).split(".")[0]
            with open(file_path, "r", encoding="utf8") as r:
                text = r.read()
            pred_entities = extractor.extract(text)
            for start, end, pred_tag in pred_entities:
                assert pred_tag == tag, [pred_tag, tag]
                pred_set.add((file_id, tag, str(start), str(end-1), text[start:end]))
            ann_path = file_path[:-4] + ".csv"
            ann_df = pd.read_csv(ann_path, dtype=str)
            
            records = ann_df.to_records(index=None)
            records = [tuple(record) for record in records if record[1]==tag]
            labe_set.update(records)
        print(list(pred_set)[0])
        print(list(labe_set)[0])

        def precision_score(y_true, y_pred, average='micro'):
            true_entities = set(y_true)
            pred_entities = set(y_pred)

            nb_correct = len(true_entities & pred_entities)
            nb_pred = len(pred_entities)

            score = nb_correct / nb_pred if nb_pred > 0 else 0

            return score

        def recall_score(y_true, y_pred, average='micro', suffix=False):
            true_entities = set(y_true)
            pred_entities = set(y_pred)

            nb_correct = len(true_entities & pred_entities)
            nb_true = len(true_entities)

            score = nb_correct / nb_true if nb_true > 0 else 0

            return score

        def f_score(y_true, y_pred, average='micro', suffix=False):
            true_entities = set(y_true)
            pred_entities = set(y_pred)

            nb_correct = len(true_entities & pred_entities)
            nb_pred = len(pred_entities)
            nb_true = len(true_entities)

            p = nb_correct / nb_pred if nb_pred > 0 else 0
            r = nb_correct / nb_true if nb_true > 0 else 0
            score = 2 * p * r / (p + r) if p + r > 0 else 0

            return score

        print(f"for {tag}: ")

        precision = precision_score(labe_set, pred_set)
        print(f"precision: {precision}")

        recall = recall_score(labe_set, pred_set)
        print(f"recall: {recall}")

        f_score = f_score(labe_set, pred_set)
        print(f"f_score: {f_score}")

        sorted_func = lambda x: sorted(list(x), key=lambda y: (int(y[0]), int(y[2]), int(y[3])))
        print("在label，不在pred", sorted_func(labe_set-pred_set))
        print("在pred, 不在label", sorted_func(pred_set-labe_set))
        print("")

    test_extractor(vx_extractor, "vx")
    test_extractor(qq_extractor, "QQ")
    test_extractor(email_extractor, "email")
        
        









