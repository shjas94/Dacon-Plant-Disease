import os
import glob
import yaml
import pickle as pkl
import numpy as np
import pandas as pd

if __name__ == "__main__":
    with open(os.path.join(os.getcwd(), 'config', 'cfg1.yml')) as f:
        cfg = yaml.safe_load(f)
    pkl_path_list = glob.glob(os.getcwd(), 'submissions', '*.pkl')
    pkl_files = [pd.read_csv(pkl_path) for pkl_path in pkl_path_list]
    with open(pkl_files[0], 'rb') as f:
        proba = pkl.load(f)
    for i in range(1, len(pkl_files)):
        with open(pkl_files[i]) as f:
            temp = pkl.load(f)
        proba += temp
    proba /= len(pkl_files)
    final_prediction = np.argmax(proba, axis=-1)

    submission = pd.read_csv(os.path.join(
        cfg['PATH']['DATA'], 'sample_submission.csv'))
    submission.iloc[:, 1] = final_prediction
    submission.to_csv(os.path.join(
        '/content', cfg['INFERENCE']['SUBMISSION_NAME']), index=False)
