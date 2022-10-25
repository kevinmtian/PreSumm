"""process raw csv for xianjinyuan data into the format of seq.in and seq.out according to slot labels
correspond to the 2022-10-15 huizong2 data
"""
import os
import pandas as pd
import re
import numpy as np

def report_labeled(golden_file, pred_file, src_file, report_csv_file):
    """merge abstract and label into one document, similar to story format, use @highlight token"""
    golden_list = []
    pred_list = []
    src_list = []
    with open(golden_file, "r") as f:
        golden_list = [x.strip() for x in f.readlines()]
    with open(pred_file, "r") as f:
        pred_list = [x.strip() for x in f.readlines()]
    with open(src_file, "r") as f:
        src_list = [x.strip() for x in f.readlines()]
    
    df_report = pd.DataFrame({
        "src": src_list,
        "label": golden_list,
        "pred": pred_list,
    }).reset_index()

    df_report.to_csv(report_csv_file, header=True, index=False)
        

if __name__ == "__main__":
    golden_file = "/data/tianmu/checkpoints/PreSumm/huizong2/story/result_test_step2000.gold"
    pred_file = "/data/tianmu/checkpoints/PreSumm/huizong2/story/result_test_step2000.candidate"
    src_file = "/data/tianmu/checkpoints/PreSumm/huizong2/story/result_test_step2000.src"
    report_csv_file = "/data/tianmu/checkpoints/PreSumm/huizong2/story/report_test_step2000.csv"
    
    report_labeled(golden_file, pred_file, src_file, report_csv_file)