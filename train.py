import os
import glob
import yaml
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from tqdm import tqdm
import argparse
import wandb

from modules.dataset import *
from modules.models import *
from modules.utils import *
from modules.schedulers import *


def get_fold(cfg, df):
    skf = StratifiedKFold(n_splits=5, shuffle=True,
                          random_state=cfg['TRAIN']['SEED'])
    skf.get_n_splits(df, df.disease_code)
    train_indices, valid_indices = [], []
    for train_index, valid_index in tqdm(skf.split(df, df.disease_code)):
        train_indices.append(train_index)
        valid_indices.append(valid_index)
    return train_indices, valid_indices


def train(cfg, train_meta, val_meta, fold_num):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    root = cfg['PATH']['DATA']
    plant_data = pd.read_csv(os.path.join(root, 'train.csv'))
    num_classes = len(plant_data['disease_code'].unique())
    torch.cuda.empty_cache()
    # wandb.init(project='Plant_Disease',
    #            group=cfg['MODEL']['MODEL_NAME'], name=cfg['TRAIN']['RUN_NAME']+'_'+str(fold_num), config=cfg)

    transforms = A.Compose([
                           A.Resize(cfg['DATA']['IMG_SIZE'][0],
                                    cfg['DATA']['IMG_SIZE'][1], always_apply=True),
                           ToTensorV2(p=1.0)
                           ])

    augmentations = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                           rotate_limit=30, p=1.0),
        A.OneOf([

            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0)
        ], p=0.5),
        A.OneOf([
            A.RandomGamma(p=1.0),
            A.ChannelDropout(p=1.0),
            A.RGBShift(p=1.0)
        ], p=0.5),
        A.OneOf([
            A.CoarseDropout(
                max_height=10, max_width=10, p=1.0),
            A.GaussianBlur(p=1.0)
        ])
    ])

    train_set = PlantDataset(
        cfg=cfg, meta=train_meta, transforms=transforms, augmentations=augmentations, mode='train')
    val_set = PlantDataset(cfg=cfg, meta=val_meta,
                           transforms=transforms, mode='valid')

    train_loader = DataLoader(
        train_set,
        batch_size=cfg['TRAIN']['BATCH_SIZE'],
        num_workers=3,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg['TRAIN']['VAL_BATCH_SIZE'],
        num_workers=3,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False
    )

    model = Encoder(encoder_name=cfg['MODEL']['MODEL_NAME'],
                    weight_name=cfg['MODEL']['WEIGHT_NAME'], num_classes=num_classes)
    model.to(device)
    criterion1, criterion2 = None, None
    if cfg['TRAIN']['MULTI_CRITERION']:
        criterion1, criterion2 = get_criterion(
            cfg['TRAIN']['CRITERION1'], cfg), get_criterion(cfg['TRAIN']['CRITERION2'], cfg)
    else:
        criterion1 = get_criterion(cfg['TRAIN']['CRITERION1'], cfg)
    max_norm = 1.
    optimizer = get_optimizer(cfg, model)
    val_criterion = get_criterion(cfg['TRAIN']['CRITERION2'], cfg)
    # Add Scheduler!!
    if cfg['TRAIN']['SCHEDULER'] == 'cosinewarmup':
        scheduler = CosineAnnealingWarmUpRestart(
            optimizer=optimizer, T_0=4, T_mult=1, eta_max=2e-4,  T_up=1, gamma=0.5)

    best_valid_loss = np.inf
    best_model = None
    # wandb.watch(model)
    early_stopping = EarlyStopping(
        patience=cfg['TRAIN']['PATIENCE'], verbose=True)

    for epoch in range(cfg['TRAIN']['EPOCH']):
        train_loss_list = []
        with tqdm(train_loader,
                  total=train_loader.__len__(),
                  unit='batch') as train_bar:
            for sample in train_bar:
                train_bar.set_description(f"Train Epoch: {epoch}")
                optimizer.zero_grad()
                images, labels = sample['image']['image'].float(
                ), sample['label'].long()
                images = images.to(device)
                labels = labels.to(device)

                if cfg['TRAIN']['MIXUP'] == True:
                    images, label_a, label_b, lam = mixup_data(images, labels,
                                                               cfg['TRAIN']['ALPHA'], use_cuda)
                    images, label_a, label_b = map(Variable, (images,
                                                              label_a, label_b))
                model.train()

                with torch.set_grad_enabled(True):
                    preds = model(images)

                    if cfg['TRAIN']['MIXUP'] == True:
                        if cfg['TRAIN']['MULTI_CRITERION']:
                            loss = lam * (0.3*criterion1(preds, label_a) + 0.7*criterion2(preds, label_a)) + \
                                (1-lam)*(0.3*criterion1(preds, label_b) +
                                         0.7*criterion2(preds, label_b))
                        else:
                            loss = lam * \
                                criterion1(preds, label_a) + \
                                (1 - lam) * criterion1(preds, label_b)
                    else:
                        if cfg['TRAIN']['MULTI_CRITERION']:
                            loss = 0.3 * \
                                criterion1(preds, labels) + 0.7 * \
                                criterion2(preds, labels)
                        else:
                            loss = criterion1(preds, labels)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm)

                    if cfg['TRAIN']['OPTIMIZER'] == 'sam' and not cfg['TRAIN']['MULTI_CRITERION'] and cfg['TRAIN']['MIXUP']:
                        optimizer.first_step(zero_grad=True)
                        second_loss = lam * \
                            criterion1(model(images), label_a) + (1 -
                                                                  lam) * criterion1(model(images), label_b)
                        second_loss.backward()
                        optimizer.second_step(zero_grad=True)
                    elif cfg['TRAIN']['OPTIMIZER'] == 'sam' and not cfg['TRAIN']['MULTI_CRITERION'] and not cfg['TRAIN']['MIXUP']:
                        optimizer.first_step(zero_grad=True)
                        second_loss = criterion1(model(images), labels)
                        second_loss.backward()
                        optimizer.second_step(zero_grad=True)
                    else:
                        optimizer.step()

                    preds = preds.cpu().detach().numpy()
                    labels = labels.cpu().detach().numpy()
                    preds = np.argmax(preds, axis=-1)

                    train_loss_list.append(loss.item())
                    train_loss = np.mean(train_loss_list)
                    learning_rate = get_lr(optimizer)
                    # wandb.log({
                    #     "Train Loss": train_loss,
                    #     "Learning Rate": learning_rate
                    # })
                    train_bar.set_postfix(train_loss=train_loss)

        valid_loss_list = []
        valid_f1_list = []
        with tqdm(val_loader,
                  total=val_loader.__len__(),
                  unit="batch") as valid_bar:
            for sample in valid_bar:
                valid_bar.set_description(f"Valid Epoch: {epoch}")
                optimizer.zero_grad()

                images, labels = sample['image']['image'].float(
                ), sample['label'].long()

                images = images.to(device)
                labels = labels.to(device)

                model.eval()
                with torch.no_grad():
                    preds = model(images)
                    # only use criterion1 for validation process
                    valid_loss = val_criterion(preds, labels)

                    preds = preds.cpu().detach().numpy()
                    labels = labels.cpu().detach().numpy()
                    preds = np.argmax(preds, axis=-1)

                    batch_f1 = f1_score(labels, preds, average='macro')
                    valid_f1_list.append(batch_f1)
                    valid_f1 = np.mean(valid_f1_list)

                    valid_loss_list.append(valid_loss.item())
                    valid_loss = np.mean(valid_loss_list)

                    valid_bar.set_postfix(valid_loss=valid_loss,
                                          valid_f1=valid_f1)
        # wandb.log({
        #     "Valid Loss": valid_loss,
        #     "Valid F1": valid_f1,
        # })
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("stopped Earlier than Expected!!!!!!!!!!!!")
            save_dir = cfg['PATH']['SAVE']
            run_name = cfg['TRAIN']['RUN_NAME']

            if not os.path.exists(os.path.join(save_dir, run_name)):
                os.mkdir(os.path.join(save_dir, run_name))
            save_path = os.path.join(save_dir, run_name) + '/'
            torch.save(
                best_model, f'{save_path}_{fold_num}_{best_valid_loss:2.4f}_epoch_{best_epoch}.pth')
            # wandb.join()
            break
        if best_valid_loss > valid_loss:
            print()
            print(
                f"Best Model Changed!!, Previous: {best_valid_loss} VS current: {valid_loss}")
            best_valid_loss = valid_loss
            best_model = model
            best_epoch = epoch
        if cfg['TRAIN']['SCHEDULER']:
            scheduler.step()
    save_dir = cfg['PATH']['SAVE']

    run_name = cfg['TRAIN']['RUN_NAME']

    if not os.path.exists(os.path.join(save_dir, run_name)):
        os.mkdir(os.path.join(save_dir, run_name))
    save_path = os.path.join(save_dir, run_name) + '/'
    torch.save(
        best_model, f'{save_path}_{fold_num}_{best_valid_loss:2.4f}_epoch_{best_epoch}.pth')
    # wandb.join()
    return best_model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="configuration file name")
    args = parser.parse_args()

    with open(os.path.join(os.getcwd(), 'config', args.config)) as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg['TRAIN']['SEED'])
    df = pd.read_csv(os.path.join(cfg['PATH']['DATA'], 'train.csv'))
    train_sets, valid_sets = get_fold(cfg, df)
    best_model = []
    for i, (train_set, valid_set) in enumerate(zip(train_sets, valid_sets)):
        best_model.append(
            train(cfg, df.iloc[train_set], df.iloc[valid_set], fold_num=i))
