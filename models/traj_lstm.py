"""
Date: 03.27.2020
Intro:
    1. Traj with LSTM;
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ST_GCN_18(nn.Module):
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

        """ func for traj """
        self.enc_xy_emb = torch.nn.Linear(2, 128, bias=True)

        self.traj_encoder = Traj_Enc_LSTM(128, 256)
        self.traj_decoder = Traj_Dec_LSTM(128, 256) # this step is a recursive process

        # self.traj_encoder = Traj_Enc_GRU()
        # self.traj_decoder = Traj_Dec_GRU()

        self.traj_emb2xy = torch.nn.Linear(256, 2, bias=True)

        """ non-linear func """
        # self.tanh = nn.Tanh()
        self.tanh = nn.Tanhshrink()
        self.tahh = nn.ReLU()

    def forward(self, traj, traj_gt, mode='train', pre_len=12):
        ''' run for traj_pred
        @param x: input KP
        @param appear: input appearance seq
        @param traj: obs traj seq
        @param traj_gt: ground-truth trajectory for teacher forcing
        @param mode: either 'train' or 'test', for decide whether teacher forcing or not
        @param hidden: initial hidden state for LSTM;
        '''

        N = traj.size()[0]

        ''' process obs traj data '''
        traj_emb = self.enc_xy_emb(traj)

        ''' init h0/c0 for lstm '''
        h0 = Variable(torch.zeros(1, N, 256)).cuda()
        c0 = Variable(torch.zeros(1, N, 256)).cuda()

        assert len(traj_emb.shape) == 3
        traj_emb = traj_emb.permute(1, 0, 2) # to [seq_len, batch, dim]
        traj_hidden = self.traj_encoder(traj_emb, (h0, c0))

        ''' process gt traj data '''
        traj_gt_emb = self.enc_xy_emb(traj_gt)
        traj_gt_emb = traj_gt_emb.permute(1, 0, 2) # to [seq_len, batch, dim]

        ''' correlation among hidden states & cell '''
        # h_n = (traj_hidden[0] * torch.squeeze(x))
        # c_n = (traj_hidden[1] * torch.squeeze(x))
        # x_hidden = self.fcn2hidden(x) # 128

        h_n = traj_hidden[0]
        c_n = traj_hidden[1]
        # h_n = torch.cat((h_n, torch.squeeze(x_hidden).unsqueeze_(0)), -1) # concate for hidden states
        # c_n = torch.cat((c_n, torch.squeeze(x_hidden).unsqueeze_(0)), -1)
        # x_hidden = self.fcn2hidden(x) # 128

        # x_cell = self.fcn2cell(x)
        # h_n = torch.squeeze(x_hidden).unsqueeze_(0) # use ST-GCN feature to initialize decoder
        # c_n = torch.squeeze(x_cell).unsqueeze_(0) # use ST-GCN feature to initialize decoder

        ''' recursively decode '''
        last_emb_xy = traj_emb[-1].clone()
        traj_out_emb_list = [] # save predicted embedding
        traj_out_xy_list = []

        if mode == 'train':
            for i in range(pre_len):
                if i == 0:
                    next_input = last_emb_xy.unsqueeze_(0)
                    out, (h_n, c_n) = self.traj_decoder(next_input, (h_n, c_n)) # unsqueeze to [1, batch, dim]
                    # out, h_n = self.traj_decoder(last_emb_xy.unsqueeze_(0), h_n) # unsqueeze to [1, batch, dim]
                    xy = self.traj_emb2xy(out)
                else:
                    ''' teacher forcing step '''
                    next_input = traj_gt_emb[i-1].clone().unsqueeze_(0)
                    out, (h_n, c_n) = self.traj_decoder(next_input, (h_n, c_n)) # teacher forcing
                    # out, h_n = self.traj_decoder(traj_gt_emb[i-1].clone().unsqueeze_(0), h_n) # teacher forcing
                    xy = self.traj_emb2xy(out)
                # traj_out_emb_list.append(xy)
                traj_out_xy_list.append(xy)

        elif mode == 'test':
            for i in range(pre_len):
                if i == 0:
                    next_input = last_emb_xy.unsqueeze_(0)
                    out, (h_n, c_n) = self.traj_decoder(next_input, (h_n, c_n)) # forget to init (h_n, c_n)
                    # out, h_n = self.traj_decoder(last_emb_xy.unsqueeze_(0), h_n) # unsqueeze to [1, batch, dim]
                    xy = self.traj_emb2xy(out)
                    out = self.enc_xy_emb(xy)
                else:
                    next_input = out
                    out, (h_n, c_n) = self.traj_decoder(next_input, (h_n, c_n)) # recursively inference
                    # out, h_n = self.traj_decoder(out, h_n) # teacher forcing
                    xy = self.traj_emb2xy(out)
                    out = self.enc_xy_emb(xy)
                traj_out_xy_list.append(xy)

        # traj_out_emb = torch.stack(traj_out_emb_list)
        # traj_out_xy = self.traj_emb2xy(traj_out_emb.view(N * pre_len, -1))
        traj_out_xy = torch.stack(traj_out_xy_list)
        traj_out_xy = torch.squeeze(traj_out_xy)

        ''' for bacth == 1'''
        if len(traj_out_xy.shape) == 2:
            traj_out_xy.unsqueeze_(1)
        traj_out_xy = traj_out_xy.permute(1, 0, 2)

        return traj_out_xy

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
