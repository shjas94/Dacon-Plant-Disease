import os
import pickle as pkl
import glob
import yaml
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from modules.dataset import *
from modules.utils import *
from modules.models import *


def inference(cfg, m, test_meta):
    # submission = pd.read_csv(os.path.join(cfg['root'], 'sample_submission.csv'))
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.cuda.empty_cache()

    model = torch.load(m).to(device)
    test_df = get_test_paths(cfg)

    transforms = A.Compose([
                           A.Resize(cfg['DATA']['IMG_SIZE'][0],
                                    cfg['DATA']['IMG_SIZE'][1], always_apply=True),
                           ToTensorV2(p=1.0)
                           ])

    test_set = PlantTestDataset(cfg, test_meta, transforms=transforms)
    test_loader = DataLoader(
        test_set,
        batch_size=cfg['INFERENCE']['BATCH_SIZE'],
        num_workers=3,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False
    )
    prediction_list = []
    model.eval()
    with torch.no_grad():
        with tqdm(test_loader,
                  total=test_loader.__len__(),
                  unit="batch") as test_bar:
            for sample in test_bar:

                images = sample['image']['image']
                images = images.to(device)
                probs = model(images)
                probs = probs.cpu().detach().numpy()

                output = np.array(probs, dtype=np.float32)
                prediction_list.extend(output)
    return prediction_list


def k_fold_ensemble(cfg, test_df, prediction_arrays):
    prediction_arrays_probs = softmax(prediction_arrays[0])
    for i in range(1, len(prediction_arrays)):
        prediction_arrays_probs += softmax(prediction_arrays[i])
    prediction_arrays_probs /= len(prediction_arrays)
    return prediction_arrays_probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="configuration file name")
    args = parser.parse_args()

    with open(os.path.join(os.getcwd(), 'config', args.config)) as f:
        cfg = yaml.safe_load(f)
    test_meta = pd.read_csv(os.path.join(cfg['PATH']['DATA'], 'test.csv'))

    model_path = glob.glob(os.path.join(
        cfg['PATH']['SAVE'], cfg['TRAIN']['RUN_NAME'], '*.pth'))
    prediction_arrays = []
    for m in model_path:
        prediction_arrays.append(inference(cfg, m, test_meta))
    prediction = k_fold_ensemble(cfg, test_meta, prediction_arrays)
    filename = os.path.join(
        os.getcwd(), 'submissions', cfg['TRAIN']['RUN_NAME']+'_proba.pkl')
    fileObject = open(filename, 'wb')
    pkl.dump(prediction, fileObject)
    fileObject.close()
    # prediction = np.argmax(prediction, axis=-1)
    # submission = pd.read_csv(os.path.join(
    #     cfg['PATH']['DATA'], 'sample_submission.csv'))
    # submission.iloc[:, 1] = prediction

    # submission.to_csv(os.path.join(
    #     os.getcwd(), 'submissions', cfg['INFERENCE']['SUBMISSION_NAME']), index=False)
