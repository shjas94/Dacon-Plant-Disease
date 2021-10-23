# 🫒 작물 병해 분류 AI 경진대회 Solution (Private 12th)

# Content

* Overall Solution
* Code Structure
* How to Train & Inference & Ensemble
  
# Overall Solution

* Model : Swin Transformer Large, Base / EfficientNetV2 M (Soft Voting Ensemble)
* Optimizer: SAM(Base optimizer = RAdam)
* Criterion: Focal Loss(gamma=2.0)
* Learning rate: 1e-04
* Weight Decay: 1e-06
* Preprocessing, Augmentations: 전처리로 Histogram Equalization 적용, 상세한 Augmentation Setting은 train.py 코드 참조
  
# Code Structure

```
.
├── DATA
├── README.md
├── config
│   ├── cfg1.yml
│   ├── cfg2.yml
│   └── cfg3.yml
├── ensemble.py
├── inference.py
├── models
├── modules
│   ├── dataset.py
│   ├── losses.py
│   ├── models.py
│   ├── optimizers.py
│   ├── schedulers.py
│   └── utils.py
├── submissions
└── train.py
```
  
# How to Train & Inference & Ensemble
* Model1 & Model2 & Model3 Train
```
$ python train.py --config cfg1.yml
$ python train.py --config cfg2.yml
$ python train.py --config cfg3.yml
```
* Model1 & Model2 & Model3 Inference
```
$ python inference.py --config cfg1.yml
$ python inference.py --config cfg2.yml
$ python inference.py --config cfg3.yml
```

* Final Ensemble
```
$ python ensemble.py
```