# Pytorch-Peek-into-the-future
Pytorch implementation of CVPR19 peek into the future. To reveal.

## Content
1. LSTM Baseline for trajectory (ADE/FDE);
2. LSTM Baseline for future activity (mAP);
3. data_utils.py for loading data in Pytorch;

## Result
0. Trained weights are in **weights** folder;
1. For trjactory, Pytorch LSTM baseline achieves 18.35/37.519 (ADE/FDE);
2. For future activity, Pytorch LSTM baseline achieves 0.1998 mAP (mean average precision)
4. Above results come from test-set;

## Data preparation
1. Follows next-prediction official github to preprocess data;
2. Simply modify path in maim\*.py and data_utils.py;

## Discussion
1. Why baselines in pytorch achieve fairly decent results ??
