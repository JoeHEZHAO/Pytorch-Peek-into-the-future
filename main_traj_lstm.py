# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

"""
Intro: Predicting trajectory sololy based on obs_traj
"""

import os, sys
import argparse
import math
import sys
from tqdm import tqdm
import numpy as np
import random
import sklearn
from sklearn.metrics import average_precision_score

import data_utils as dl
from data_utils import get_data_feed
import torch
import torch.nn as nn
from other_model.traj_lstm import *

parser = argparse.ArgumentParser()

# inputs and outputs
# parser.add_argument("-prepropath", type=str, default='/home/zhufl/Data2/next-prediction/actev_preprocess')
parser.add_argument("-prepropath", type=str, default='/home/zhufl/Data2/next-prediction/actev_preprocess')
parser.add_argument("-outbasepath", type=str, default='/home/zhufl/Data2/motion_graph/next-models/actev_single_model',
                    help="full path will be outbasepath/modelname/runId")
parser.add_argument("-modelname", type=str, default='model')
parser.add_argument("--runId", type=int, default=2,
                    help="used for run the same model multiple times")

# ---- gpu stuff. Now only one gpu is used
parser.add_argument("--gpuid", default=0, type=int)

parser.add_argument("--load", action="store_true",
                    default=False, help="whether to load existing model")
parser.add_argument("--load_best", action="store_true",
                    default=False, help="whether to load the best model")
# use for pre-trained model
parser.add_argument("--load_from", type=str, default=None)

# ------------- experiment settings
parser.add_argument("--obs_len", type=int, default=8)
parser.add_argument("--pred_len", type=int, default=12)
parser.add_argument("--is_actev", action="store_true",
                    help="is actev/virat dataset, has activity info")

# ------------------- basic model parameters
parser.add_argument("--emb_size", type=int, default=128)
parser.add_argument("--enc_hidden_size", type=int,
                    default=256, help="hidden size for rnn")
parser.add_argument("--dec_hidden_size", type=int,
                    default=256, help="hidden size for rnn")
parser.add_argument("--activation_func", type=str,
                    default="tanh", help="relu/lrelu/tanh")

# ---- multi decoder
parser.add_argument("--multi_decoder", action="store_true")

# ----------- add person appearance features
# parser.add_argument("--person_feat_path", type=str, default='/home/zhufl/Data2/next-prediction/next-data/actev_personboxfeat')
parser.add_argument("--person_feat_path", type=str, default='/home/zhufl/Data2/next-prediction/next-data/actev_personboxfeat')
parser.add_argument("--person_feat_dim", type=int, default=256)
parser.add_argument("--person_h", type=int, default=9,
                    help="roi align resize to feature size")
parser.add_argument("--person_w", type=int, default=5,
                    help="roi align resize to feature size")

# ---------------- other boxes
parser.add_argument("--random_other", action="store_true",
                    help="randomize top k other boxes")
parser.add_argument("--max_other", type=int, default=15,
                    help="maximum number of other box")
parser.add_argument("--box_emb_size", type=int, default=64)

# ---------- person pose features
parser.add_argument("--add_kp", action="store_true")
parser.add_argument("--kp_size", default=17, type=int)

# --------- scene features
parser.add_argument("--scene_conv_kernel", default=3, type=int)
parser.add_argument("--scene_h", default=36, type=int)
parser.add_argument("--scene_w", default=64, type=int)
parser.add_argument("--scene_class", default=11, type=int)
parser.add_argument("--scene_conv_dim", default=64, type=int)
parser.add_argument("--pool_scale_idx", default=0, type=int)

#  --------- activity
parser.add_argument("--add_activity", action="store_true")

#  --------- loss weight
parser.add_argument("--act_loss_weight", default=1.0, type=float)
parser.add_argument("--grid_loss_weight", default=0.1, type=float)
parser.add_argument("--traj_class_loss_weight", default=1.0, type=float)

# ---------------------------- training hparam
parser.add_argument("--save_period", type=int, default=300,
                    help="num steps to save model and eval")
parser.add_argument("--batch_size", type=int, default=256)
# num_step will be num_example/batch_size * epoch
parser.add_argument("--num_epochs", type=int, default=100)
# drop out rate
parser.add_argument("--keep_prob", default=0.7, type=float,
                    help="1.0 - drop out rate")
# l2 weight decay rate
parser.add_argument("--wd", default=0.0001, type=float,
                    help="l2 weight decay loss")
parser.add_argument("--clip_gradient_norm", default=10, type=float,
                    help="gradient clipping")
parser.add_argument("--optimizer", default="adadelta",
                    help="momentum|adadelta|adam")
parser.add_argument("--learning_rate_decay", default=0.95,
                    type=float, help="learning rate decay")
parser.add_argument("--num_epoch_per_decay", default=2.0,
                    type=float, help="how epoch after which lr decay")
parser.add_argument("--init_lr", default=0.2, type=float,
                    help="Start learning rate")
parser.add_argument("--emb_lr", type=float, default=1.0,
                    help="learning scaling factor for emb variables")

def main(args):
    """Run training."""
    val_perf = []  # summary of validation performance, and the training loss

    train_data, train_vid2name = dl.read_data(args, "train")
    val_data, val_vid2name = dl.read_data(args, "val")
    test_data, test_vid2name = dl.read_data(args, "test")
    train_vid2name, val_vid2name = train_vid2name.item(), val_vid2name.item()

    args.train_num_examples = train_data.num_examples

    num_steps = int(math.ceil(train_data.num_examples /
                              float(args.batch_size)))*args.num_epochs

    num_steps_test = int(math.ceil(test_data.num_examples / float(args.batch_size)))

    num_steps_val = int(math.ceil(val_data.num_examples / float(args.batch_size)))

    """ load st-gcn model """
    model = ST_GCN_18(
            in_channels=2,
            num_class=30,
    ).cuda()

    """ init learnable weights """
    model = model.apply(weights_init)

    """ init multi-class loss func """
    # criterion = nn.CrossEntropyLoss()
    multi_criterion = nn.BCEWithLogitsLoss()
    MSE_criterion = nn.MSELoss(reduction='mean')

    """ init optim """
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    """ init best ade/fde """
    best_ade, best_fde = 999, 999

    """ Loop batch """
    for idx, batch in enumerate(tqdm(train_data.get_batches(args.batch_size,
                                             num_steps=num_steps), total=num_steps, ascii=True)):

        batch_idx = batch[0]
        batch_train = batch[1]

        data, other_boxes_seq = get_data_feed(batch_train, data_type='train', N=args.batch_size)

        """ process input traj data rel value """
        input_traj_tensor = np.stack(data['obs_traj_rel'])
        input_traj_tensor = torch.from_numpy(input_traj_tensor)
        input_traj_gt_tensor = np.stack(data['pred_traj_rel'])
        input_traj_gt_tensor = torch.from_numpy(input_traj_gt_tensor)

        """ process input traj data abs value """
        input_traj_abs_tensor = np.stack(data['obs_traj'])
        input_traj_abs_tensor = torch.from_numpy(input_traj_abs_tensor)
        input_traj_gt_abs_tensor = np.stack(data['pred_traj'])
        input_traj_gt_abs_tensor = torch.from_numpy(input_traj_gt_abs_tensor)

        # convert to gpu
        # gt = gt.cuda()
        # input_tensor = input_tensor.cuda()
        input_traj_tensor = Variable(input_traj_tensor).cuda()
        input_traj_gt_tensor = Variable(input_traj_gt_tensor).cuda()
        input_traj_abs_tensor = Variable(input_traj_abs_tensor).cuda()
        input_traj_gt_abs_tensor = Variable(input_traj_gt_abs_tensor).cuda()

        """ zero out the gradient """
        optimizer.zero_grad()

        # produce output given inputs
        pred_traj = model(input_traj_tensor, input_traj_gt_tensor)

        # produce loss given out & gt
        diff = pred_traj - input_traj_gt_tensor
        loss_traj = torch.pow(diff, 2)
        loss_traj = loss_traj.mean()

        # perf optim
        loss_traj.backward()
        optimizer.step()

        """ run evaluation every 10 steps """
        if (idx + 1) % 50 == 0:
            with torch.no_grad():
            # for batch in tqdm(val_data.get_batches(args.batch_size,
                l2dis_list = []
                for batch in tqdm(test_data.get_batches(args.batch_size,
                                num_steps=num_steps_test), total=num_steps_test, ascii=True):
                # for batch in tqdm(val_data.get_batches(args.batch_size,
                #                 num_steps=num_steps_val), total=num_steps_val, ascii=True):

                    batch_idx = batch[0]
                    batch_val = batch[1]

                    data, other_boxes_seq = get_data_feed(batch_val, data_type='test', N=args.batch_size)
                    # data, other_boxes_seq = get_data_feed(batch_val, data_type='val', N=args.batch_size)

                    # process input traj data
                    input_traj_tensor = np.stack(data['obs_traj_rel'])
                    input_traj_tensor = torch.from_numpy(input_traj_tensor)
                    input_traj_gt_tensor = np.stack(data['pred_traj_rel'])
                    input_traj_gt_tensor = torch.from_numpy(input_traj_gt_tensor)

                    # convert to gpu
                    input_traj_tensor = Variable(input_traj_tensor).cuda()
                    input_traj_gt_tensor = Variable(input_traj_gt_tensor).cuda()

                    # produce output given inputs
                    pred_traj = model(input_traj_tensor, input_traj_gt_tensor, mode='test', pre_len=12)

                    ''' process traj for ade/fde'''
                    traj_obs_abs = np.stack(data['obs_traj'])
                    pred_traj_rel = pred_traj.cpu().data.numpy()
                    pred_traj_abs = relative_to_abs(pred_traj_rel, traj_obs_abs[:, -1, :])

                    diff = pred_traj_abs - np.stack(data['pred_traj']) # pre_traj is abs value
                    diff = diff**2
                    diff = np.sqrt(np.sum(diff, axis=2)) # [batch, pre_len]
                    l2dis_list.append(diff)

                ''' perf eval on ade/fde '''
                ade = [t for o in l2dis_list for t in o]
                fde = [t[-1] for o in l2dis_list for t in o]
                print("trajectory ADE/FDE: {}/{}".format(np.mean(ade), np.mean(fde)))

                if np.mean(ade) < best_ade or np.mean(fde) < best_fde:

                    best_ade, best_fde = np.mean(ade), np.mean(fde)

                    save_path = './traj_lstm/test_set/'
                    if os.path.isdir(save_path) is False:
                        os.mkdir(save_path)

                    torch.save({
                                'epoch': idx,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss_traj': loss_traj,
                                'ade' : ade,
                                'fde' : fde,
                    }, os.path.join(save_path, 'traj_lstm'))

def relative_to_abs(rel_traj, start_pos):
  """Relative x,y to absolute x,y coordinates.

  Args:
    rel_traj: numpy array [Batch, T,2]
    start_pos: [Batch, 2]
  Returns:
    abs_traj: [Batch, T,2]
  """

  # batch, seq_len, 2
  # the relative xy cumulated across time first
  displacement = np.cumsum(rel_traj, axis=1)
  abs_traj = displacement + start_pos[:, np.newaxis, :]  # [1,2]
  return abs_traj

def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv1d') != -1:
        model.weight.data.normal_(0.0, 0.02)
        if model.bias is not None:
            model.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        model.weight.data.normal_(0.0, 0.02)
        if model.bias is not None:
            model.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)

if __name__ == "__main__":
    arguments = parser.parse_args()
    arguments.is_train = True
    arguments.is_test = False
    arguments = dl.process_args(arguments)

    main(arguments)
