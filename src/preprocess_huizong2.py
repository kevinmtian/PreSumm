"""process raw csv for xianjinyuan data into the format of seq.in and seq.out according to slot labels
correspond to the 2022-10-15 huizong2 data
"""
import os
import pandas as pd
import re
import numpy as np

def is_contain_chinese(x):
    """return true if x contains any chinese characters"""
    if len(re.findall(r'[\u4e00-\u9fff]+', x)):
        return True
    return False

def process_csv_labeled(csv_file, output_dir, map_dir, label_token="@highlight"):
    """merge abstract and label into one document, similar to story format, use @highlight token"""
    np.random.seed(42)

    df = pd.read_csv(csv_file, header=0).reset_index()
    # generate label and raw text (processed text should be in lowercase and seperated by single spaces)
    r_base_on = "base on\s?\("
    r_with = "\)\s?with\s?\("
    r_solve = "\)\s?solve\s?\("
    r_use = "\)\s?use\s?\("
    r_enable = "\)\s?enable\s?\("
    r_for = "\)\s?for\s?\("
    r_with2 = "\)\s?with\s?\("

    def strip_helper_(x):
        # x = x.replace(",", " , ").replace(".", " . ").replace("?", " ? ").replace(";", " ; ").replace("!", " ! ")
        return " ".join(w.strip().lower() for w in x.strip().split())
    
    count_invalid_label = 0
    train_list = []
    test_list = []
    valid_list = []
    for i in range(df.shape[0]):
        sample_id = i
        # print(sample_id)
        if pd.isnull(df.loc[i, "abstract"]) or pd.isnull(df.loc[i, "template"]):
            continue
        
        raw_text = strip_helper_(df.loc[i, "abstract"].strip())
        labeled_text = strip_helper_(df.loc[i, "template"].strip())
        labeled_text = labeled_text.replace("（", "(").replace("）", ")")
        regex_template = "({})(.*)({})(.*)({})(.*)({})(.*)({})(.*)({})(.*)({})(.*)".format(
            r_base_on,
            r_with,
            r_solve,
            r_use,
            r_enable,
            r_for,
            r_with2,
        )
        
        template = re.search(regex_template, labeled_text)
        if template is None:
            count_invalid_label += 1
            print(f"num invalid label {count_invalid_label}")
            # print(labeled_text)
            # print(regex_template)
            continue

        s_base_on = strip_helper_(template.group(2))
        s_with = strip_helper_(template.group(4))
        s_solve = strip_helper_(template.group(6))
        s_use = strip_helper_(template.group(8))
        s_enable = strip_helper_(template.group(10))
        s_for = strip_helper_(template.group(12))
        s_with2 = strip_helper_(template.group(14))
        if s_with2.endswith(")"):
            s_with2 = s_with2[:-1]
        
        # xx = [s_base_on, s_with, s_solve, s_use, s_enable, s_for, s_with2]
        xx = [s_base_on]
        # for simplicity, let's simply extract the "based on" part so far!
        record = []
        for x in xx:
            x = x.strip()
            if is_contain_chinese(x):
                continue
            if not x.strip():
                continue
            record.append(x)
        if len(record) and len(raw_text.strip()):
            x = np.random.rand()
            if x < 0.7:
                train_list.append("sample_{}.story".format(sample_id))
            else:
                test_list.append("sample_{}.story".format(sample_id))
                valid_list.append("sample_{}.story".format(sample_id))

            with open(os.path.join(output_dir, "sample_{}.story".format(sample_id)), 'w', encoding='utf-8') as f_output:
                f_output.write(raw_text + '\n')
                for label in record:
                    f_output.write('\n')
                    f_output.write(label_token + '\n')
                    f_output.write('\n')
                    f_output.write(label + '\n')

    with open(os.path.join(map_dir, "mapping_train.txt"), "w", encoding='utf-8') as f:
        for x in train_list:
            f.write(x + "\n")

    with open(os.path.join(map_dir, "mapping_test.txt"), "w", encoding='utf-8') as f:
        for x in test_list:
            f.write(x + "\n")

    with open(os.path.join(map_dir, "mapping_valid.txt"), "w", encoding='utf-8') as f:
        for x in valid_list:
            f.write(x + "\n")

if __name__ == "__main__":
    csv_file = "/data/tianmu/data/meta_review/template/huizong2.csv"
    output_dir = "/data/tianmu/data/PreSumm/huizong2/raw_stories"
    # train, test, dev split mapping dir
    map_dir = "/data/tianmu/data/PreSumm/huizong2/story_map"

    process_csv_labeled(
        csv_file=csv_file,
        output_dir=output_dir,
        map_dir=map_dir,
        label_token="@highlight",
    )