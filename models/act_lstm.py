"""
Date: 03.31.2020
Intro:
    1. Replicate the performance of tensorflow code;
        a.) KP --> emb --> LSTM
        b.) appear --> LSTM
        c.) other_box --> emb --> LSTM
        d.) grid_cls --> LSTM
        e.) grid_target --> LSTM
        f.) traj (rel) --> LSTM
        g.) scene --> Conv
        h.) other_box_geo --> emb --> LSTM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Next_Pred(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_cfg (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self,
                 in_channels,
                 num_class,
                 **kwargs):
        super().__init__()

        """ emb for KP  """
        self.enc_kp_emb = torch.nn.Linear(34, 128, bias=True) # 2 x 17 keypoints

        """ emb for traj """
        # self.enc_traj_emb = torch.nn.Linear(2, 256, bias=True)

        """ emb for grid class """
        # self.enc_grid_cls_emb = torch.nn.Embedding(576, 256)

        """ emb for grid target """
        # self.enc_grid_target_emb = torch.nn.Linear(2, 256, bias=True)

        """ emb for other_box class """
        # self.enc_otherbox_cls_emb = torch.nn.Linear(10, 256, bias=True)

        """ emb for other_box geo """
        # self.enc_otherbox_geo_emb = torch.nn.Linear(4, 256, bias=True)

        """ lstm for appear """
        self.appr_lstm = nn.LSTM(256, 256, dropout=0.5) # input_dim, hidden_dim

        """ lstm for traj """
        # self.traj_lstm = nn.LSTM(256, 256, dropout=0.5) # input_dim, hidden_dim

        """ lstm for kp """
        self.kp_lstm = nn.LSTM(128, 256, dropout=0.5)

        """ lstm for grid clss & grid """
        # self.grid_lstm = nn.LSTM(512, 256, dropout=0.5)

        """ lstm for other box cls & geo """
        # self.otherbox_lstm = nn.LSTM(512 * 15, 256, dropout=0.5)

        """ classifier for final result """
        # self.classifier1 = nn.Linear(256 * 5, 512, bias=True) # map concated to final class
        # self.classifier2 = nn.Linear(512, 128, bias=True)
        # self.classifier3 = nn.Linear(128, 30, bias=False)
        self.classifier = nn.Linear(256 * 2, 30, bias=False)

        """ declare non-linearity """
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, kp, appr, mode='train', pre_len=12):
        ''' run for traj_pred
        @param x: input KP
        @param appear: input appearance seq
        @param traj: obs traj seq
        @param traj_gt: ground-truth trajectory for teacher forcing
        @param mode: either 'train' or 'test', for decide whether teacher forcing or not
        @param hidden: initial hidden state for LSTM;
        '''

        # get batch size
        N = kp.size()[0]

        # perf embedding
        kp_emb = self.tanh(self.enc_kp_emb(kp))
        appr = appr
        # traj_input_emb = self.tanh(self.enc_traj_emb(traj))
        # otherbox_cls_emb = self.tanh(self.enc_otherbox_cls_emb(otherbox_cls))
        # otherbox_geo_emb = self.tanh(self.enc_otherbox_geo_emb(otherbox_geo))
        # grid_cls_emb = self.tanh(torch.squeeze(self.enc_grid_cls_emb(grid_cls)))
        # grid_target_emb = self.tanh(self.enc_grid_target_emb(grid_target))

        # perf lstm
        kp_out, (kp_last_state, kp_last_cell) = self.kp_lstm(kp_emb.permute(1, 0, 2))
        appr_out, (appr_last_state, appr_last_cell) = self.appr_lstm(appr.permute(1, 0, 2))
        # traj_out, (traj_last_state, traj_last_cell) = self.traj_lstm(traj_input_emb.permute(1, 0, 2))
        # otherbox_out, (otherbox_last_state, otherbox_last_cell) = self.otherbox_lstm(\
        #                                     torch.cat((otherbox_cls_emb, otherbox_geo_emb), -1).view(N, 8, -1).permute(1, 0, 2))
        # grid_out, (grid_last_state, grid_last_cell) = self.grid_lstm(\
        #                                 torch.cat((grid_cls_emb, grid_target_emb), -1).permute(1, 0, 2))

        # x = self.tanh(self.classifier1(torch.cat((kp_last_state[0], appr_last_state[0], traj_last_state[0],\
        #                                   otherbox_last_state[0], grid_last_state[0]), 1)))
        # x = self.tanh(self.classifier2(x))
        x = self.classifier(torch.cat((kp_last_state[0], appr_last_state[0]), 1))
        return x

''' verson 1: only attention on avg_pooled ST-GCN feature '''
def focal_attention(query, context, use_sigmoid=False, scope=None):
    """ Focal attention layer

    Args:
        query: [N, dim1]
        context: [N, num_feat, T, dim2]
        use_sigmoid: use sigmoid instead of softmax
        scope: variable scope
    """
    query, context = torch.squeeze(query), torch.squeeze(context)
    _, d = query.shape
    _, d2 = context.shape

    assert d == d2

    # query_aug = query.unsqueeze_(1).unsqueeze_(1).repeat(1, K, T, 1) # [N, K, T, d]

    # cosine simi
    with torch.no_grad():
        qn = torch.norm(query, p=2, dim=-1)
    # query_aug_norm = query.div(qn.expand_as(query))
    query_aug_norm = query.div(qn.unsqueeze_(-1).expand_as(query))

    with torch.no_grad():
        cn = torch.norm(context, p=2, dim=-1)
    context_norm = context.div(cn.unsqueeze_(-1).expand_as(context))

    # a_logits = torch.sum((query_aug_norm * context_norm), 1)
    # a_logits_maxed = torch.max(a_logits, 2) # [N, K]
    # a_logits = torch.softmax((query_aug_norm * context_norm), 1)
    a_logits = query_aug_norm * context_norm

    # attended_context = softsel(context, a_logits, use_sigmoid=use_sigmoid)
    attended_context = a_logits

    return attended_context

def softsel(target, logits, use_sigmoid=False, scope=None):
    """Apply attention weights."""

    if use_sigmoid:
        a = torch.sigmoid(logits)
    else:
        a = torch.softmax(logits, 0)  # shape is the same
    # target_rank = len(target.get_shape().as_list())
    # [N,M,JX,JQ,2d] elem* [N,M,JX,JQ,1]
    # second last dim
    # return torch.sum(a.unsqueeze_(-1)*target, target_rank-2)
    return a.unsqueeze_(-1)*target


class Traj_Dec_LSTM(torch.nn.Module):
    def __init__(self, emb_dim=256, hidden_dim=128):
        super(Traj_Dec_LSTM, self).__init__()

        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.dec_rnn = torch.nn.LSTM(self.emb_dim, self.hidden_dim, 1, dropout=0.7)

    def forward(self, traj_data, hidden):
        """ run enc lstm to traj
        @param traj_data: trajectory with shape [b, len, emb=128]
        """
        out, hidden = self.dec_rnn(traj_data, hidden)
        return out, hidden

class Traj_Enc_LSTM(torch.nn.Module):
    def __init__(self, emb_dim=128, hidden_dim=128):
        super(Traj_Enc_LSTM, self).__init__()

        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.enc_rnn = torch.nn.LSTM(self.emb_dim, self.hidden_dim, 1, dropout=0.7)

    def forward(self, traj_data, hidden):
        """ run enc lstm to traj
        @param traj_data: trajectory with shape [b, len, xy=2]
        """
        out, hidden = self.enc_rnn(traj_data, hidden)
        return hidden # [h_n, c_n]
