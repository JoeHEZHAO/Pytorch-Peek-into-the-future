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

"""Train person prediction model.
See README for running instructions.

Date: 2020.03.31
Intro: replicate tensorflow implementation into pytorch;
Goal : recover the reported accuracy of tensorflow code;
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
import torchnet.meter as meter
import operator

import data_utils as dl
from data_utils import get_data_feed
import torch
import torch.nn as nn
from models.act_lstm import Next_Pred

parser = argparse.ArgumentParser()

# inputs and outputs
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
parser.add_argument("--batch_size", type=int, default=64)
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

activity2id = {
    "BG": 0,  # background
    "activity_walking": 1,
    "activity_standing": 2,
    "activity_carrying": 3,
    "activity_gesturing": 4,
    "Closing": 5,
    "Opening": 6,
    "Interacts": 7,
    "Exiting": 8,
    "Entering": 9,
    "Talking": 10,
    "Transport_HeavyCarry": 11,
    "Unloading": 12,
    "Pull": 13,
    "Loading": 14,
    "Open_Trunk": 15,
    "Closing_Trunk": 16,
    "Riding": 17,
    "specialized_texting_phone": 18,
    "Person_Person_Interaction": 19,
    "specialized_talking_phone": 20,
    "activity_running": 21,
    "PickUp": 22,
    "specialized_using_tool": 23,
    "SetDown": 24,
    "activity_crouching": 25,
    "activity_sitting": 26,
    "Object_Transfer": 27,
    "Push": 28,
    "PickUp_Person_Vehicle": 29,
}

def main(args):
    """Run training."""
    val_perf = []  # summary of validation performance, and the training loss

    test_data, test_vid2name = dl.read_data(args, "test")

    num_steps_test = int(math.ceil(test_data.num_examples / float(args.batch_size)))

    """ load st-gcn model """
    model = Next_Pred(
            in_channels=2,
            num_class=30,
    ).cuda()

    """ init learnable weights """
    saved_file = torch.load('./weights/act_lstm.pth')
    model.load_state_dict(saved_file['model_state_dict'])
    model.eval()
    print(model)

    with torch.no_grad():
    # for batch in tqdm(val_data.get_batches(args.batch_size,
        gt_list = []
        score_list = []
        mrt = meter.mAPMeter()

        ''' init data structure for tensorflow mAP eval '''
        future_act_scores = {actid: [] for actid in activity2id.values()}
        future_act_labels = {actid: [] for actid in activity2id.values()}
        act_ap = None

        for batch in tqdm(test_data.get_batches(args.batch_size,
                        num_steps=num_steps_test, shuffle=False, full=True), total=num_steps_test, ascii=True):

            batch_idx = batch[0]
            batch_val = batch[1]

            data, other_boxes_seq = get_data_feed(batch_val, data_type='test', N=args.batch_size)

            # process gt data
            gt = data['future_activity']
            labels = [torch.zeros(30).scatter_(0, torch.tensor(x), 1.) for x in gt]
            gt = torch.stack(labels, 0)

            # process kp data # [batch, seq, 34]
            input_tensor = np.stack(data['obs_kp_rel'])
            input_tensor = torch.from_numpy(input_tensor)
            input_tensor = input_tensor.view(args.batch_size, args.obs_len, -1)

            # process appear data ==> [batch, seq, dim]
            input_appear_tensor = torch.from_numpy(np.mean(data['obs_person_feat'], (2, 3)))

            # process obs_other_box_class data ==> [batch, seq, 15, 10]
            other_box_cls = np.stack(data['obs_other_box_class'])
            other_box_cls = torch.from_numpy(other_box_cls)
            # other_box_feat = other_box_feat.unsqueeze(-1).permute(0, 3, 1, 2, -1)

            # process obs_other_box_geo data ==> [batch, seq, 15, 4]
            other_box_geo = np.stack(data['obs_other_box'])
            other_box_geo = torch.from_numpy(other_box_geo)

            # process grid class data ==> [batch, seq, 1]
            obs_grid_cls = np.stack(data['obs_grid_class'])[:, 0, :].astype('long')
            obs_grid_cls = np.reshape(obs_grid_cls, (args.batch_size, args.obs_len, 1))
            obs_grid_cls = torch.from_numpy(obs_grid_cls)
            # labels = [torch.zeros(576).scatter_(0, torch.tensor(x), 1.) for x in obs_grid_cls]
            # obs_grid_cls = torch.stack(labels, 0).view(args.batch_size, args.obs_len, -1)

            # process grid target data ==> [batch, seq, 4]
            obs_grid_target = np.stack(data['obs_grid_target'])[:, 0]
            obs_grid_target = torch.from_numpy(obs_grid_target)

            # process traj data ==> [batch, seq, 2]
            input_traj_tensor = np.stack(data['obs_traj_rel'])
            input_traj_tensor = torch.from_numpy(input_traj_tensor)

            # process obs otherbox msoe adj
            # other_box_msoe_adj = data['obs_other_box_msoeadj']
            # other_box_msoe_adj_vert = np.transpose(other_box_msoe_adj, (0, 1, 3, 2))
            # other_box_msoe_adj_hori = np.transpose(other_box_msoe_adj, (0, 1, 3, 2))
            # other_box_msoe_adj_ext = np.zeros((args.batch_size, args.obs_len, 4, 16, 16))

            # assume last index is target/ped, 0-15 is traffic-objs
            # other_box_msoe_adj_ext[:,:,:, :-1, -1] = other_box_msoe_adj_vert
            # other_box_msoe_adj_ext[:,:,:, -1, :-1] = other_box_msoe_adj_hori
            # other_box_msoe_adj_ext = torch.tensor(other_box_msoe_adj_ext.astype(np.float32), requires_grad=False)  # Form to [B, T, num_adj, 16, 16]

            ''' obtain perseon-object adjacency matrix '''
            # convert to gpu
            gt = gt.cuda()
            input_tensor = input_tensor.cuda()
            input_appear_tensor = input_appear_tensor.cuda()
            other_box_cls = other_box_cls.cuda()
            other_box_geo = other_box_geo.cuda()
            obs_grid_cls = obs_grid_cls.cuda()
            obs_grid_target = obs_grid_target.cuda()
            input_traj_tensor = input_traj_tensor.cuda()

            out = model(input_tensor, input_appear_tensor, \
                mode='train')
            mrt.add(out, gt)

        print("Average Precision is {}".format(mrt.value()))

def compute_ap(lists):
  """Compute Average Precision."""
  lists.sort(key=operator.itemgetter("score"), reverse=True)
  rels = 0
  rank = 0
  score = 0.0
  for one in lists:
    rank += 1
    if one["label"] == 1:
      rels += 1
      score += rels/float(rank)
  if rels != 0:
    score /= float(rels)
  return score

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
    elif classname.find('Linear') != -1:
        model.weight.data.normal_(0.0, 0.1)
        if model.bias is not None:
            model.bias.data.fill_(0)

if __name__ == "__main__":
    arguments = parser.parse_args()
    arguments.is_train = True
    arguments.is_test = False
    arguments = dl.process_args(arguments)
    main(arguments)
