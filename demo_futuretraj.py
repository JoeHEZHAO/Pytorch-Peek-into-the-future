###############################
# Demo of trajectory estimation
#
# Base on pure traj_lstm approach
###############################

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
from models.traj_lstm import *

import PIL
from PIL import ImageDraw
from IPython.display import display
import matplotlib.pyplot as plt
import cv2
from skimage.transform import resize

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
parser.add_argument("--batch_size", type=int, default=1)
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
parser.add_argument("--frame_root", type=str, default='/home/zhufl/Data2/next-prediction-v2/code/prepare_data/actev_all_videos_flow')

parser.add_argument("--test_output_path", type=str, default='/home/zhufl/GCN_video/mmskeleton/traj_output_plot')

def main(args):

    ''' init dataset '''
    test_data, test_vid2name = dl.read_data(args, "test")
    num_steps_test = int(math.ceil(test_data.num_examples / float(args.batch_size)))

    ''' init pretrained model '''
    model = Next_Pred(
            in_channels=2,
            num_class=30,
    ).cuda()

    saved_file = torch.load('./weights/traj_lstm.pth')
    model.load_state_dict(saved_file['model_state_dict'])
    model.eval()
    print(model)

    ''' process test data one-by-one '''
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
            pred_traj_gt = np.stack(data['pred_traj']) # abs format pred trajectory

            ''' recover pred_traj_rel to abs  '''
            pred_traj_rel = pred_traj.cpu().data.numpy()
            pred_traj_abs = relative_to_abs(pred_traj_rel, traj_obs_abs[:, -1, :])

            diff = pred_traj_abs - np.stack(data['pred_traj']) # pre_traj is abs value
            diff = diff**2
            diff = np.sqrt(np.sum(diff, axis=2)) # [batch, pre_len]
            l2dis_list.append(diff)

            ''' comment out if not visualziation '''
            ''' start visualization process '''
            video_folder = data['traj_key'][0].split('_')[:-2]
            video_folder = '_'.join(video_folder)
            frame = 'img_' + str(data['obs_frameidx'][0][-1]).zfill(8) + '.jpeg'
            frame_path = os.path.join(args.frame_root, video_folder, frame)

            if not os.path.exists(frame_path):
                frame = 'img_' + str(data['obs_frameidx'][0][-1]).zfill(8) + '.jpg' # use last frame to demostrate
                frame_path = os.path.join(args.frame_root, video_folder, frame)

            ''' recover to abs absolute '''
            img = PIL.Image.open(frame_path)
            img_draw = ImageDraw.Draw(img)
            img_draw.line(np.squeeze(traj_obs_abs), fill ="cyan", width = 6) # plot observed gt
            img_draw.line(np.squeeze(pred_traj_gt), fill ="red", width = 6) # plot gt future trajectory
            img_draw.line(np.squeeze(pred_traj_abs), fill = (0, 255, 0, 0), width = 15) # plot predicted future trajectory

            if not os.path.exists(os.path.join(args.test_output_path, video_folder)):
                os.mkdir(os.path.join(args.test_output_path, video_folder))
            img.save(os.path.join(args.test_output_path, video_folder, frame))

        ''' perf eval on ade/fde '''
        ade = [t for o in l2dis_list for t in o]
        fde = [t[-1] for o in l2dis_list for t in o]
        print("trajectory ADE/FDE: {}/{}".format(np.mean(ade), np.mean(fde)))

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

if __name__ == "__main__":
    arguments = parser.parse_args()
    arguments.is_train = True
    arguments.is_test = False
    arguments = dl.process_args(arguments)

    main(arguments)
