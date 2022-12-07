import torch
import torch.nn as nn
import math
import numpy as np


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    # nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def fc_init(fc):
    nn.init.xavier_normal_(fc.weight)
    nn.init.constant_(fc.bias, 0)


class PositionalEncoding(nn.Module):

    def __init__(self, channel, joint_num, time_len, domain):
        super(PositionalEncoding, self).__init__()
        self.joint_num = joint_num
        self.time_len = time_len

        self.domain = domain

        if domain == "temporal":
            # temporal embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(t)
        elif domain == "spatial":
            # spatial embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(j_id)

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()
        # pe = position/position.max()*2 -1
        # pe = pe.view(time_len, joint_num).unsqueeze(0).unsqueeze(0)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(self.time_len * self.joint_num, channel)

        div_term = torch.exp(torch.arange(0, channel, 2).float() *
                             -(math.log(10000.0) / channel))  # channel//2
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.view(time_len, joint_num, channel).permute(2, 0, 1).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):  # nctv
        x = x + self.pe[:, :, :x.size(2)]
        return x


class STAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels, num_subset=3, num_node=25, num_frame=32,
                 kernel_size=1, stride=1, glo_reg_s=True, att_s=True, glo_reg_t=True, att_t=True,
                 use_temporal_att=True, use_spatial_att=True, attentiondrop=0, use_pes=True, use_pet=True):
        super(STAttentionBlock, self).__init__()
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_subset = num_subset
        self.glo_reg_s = glo_reg_s
        self.att_s = att_s
        self.glo_reg_t = glo_reg_t
        self.att_t = att_t
        self.use_pes = use_pes
        self.use_pet = use_pet

        pad = int((kernel_size - 1) / 2)
        self.use_spatial_att = use_spatial_att
        if use_spatial_att:
            atts = torch.zeros((1, num_subset, num_node, num_node))
            self.register_buffer('atts', atts)
            self.pes = PositionalEncoding(in_channels, num_node, num_frame, 'spatial')
            self.ff_nets = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if att_s:
                self.in_nets = nn.Conv2d(in_channels, 2 * num_subset * inter_channels, 1, bias=True)
                self.alphas = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
            if glo_reg_s:
                self.attention0s = nn.Parameter(torch.ones(1, num_subset, num_node, num_node) / num_node,
                                                requires_grad=True)

            self.out_nets = nn.Sequential(
                nn.Conv2d(in_channels * num_subset, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.out_nets = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1), bias=True, stride=1),
                nn.BatchNorm2d(out_channels),
            )
        self.use_temporal_att = use_temporal_att
        if use_temporal_att:
            attt = torch.zeros((1, num_subset, num_frame, num_frame))
            self.register_buffer('attt', attt)
            self.pet = PositionalEncoding(out_channels, num_node, num_frame, 'temporal')
            self.ff_nett = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (kernel_size, 1), (stride, 1), padding=(pad, 0), bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if att_t:
                self.in_nett = nn.Conv2d(out_channels, 2 * num_subset * inter_channels, 1, bias=True)
                self.alphat = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
            if glo_reg_t:
                self.attention0t = nn.Parameter(torch.zeros(1, num_subset, num_frame, num_frame) + torch.eye(num_frame),
                                                requires_grad=True)
            self.out_nett = nn.Sequential(
                nn.Conv2d(out_channels * num_subset, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.out_nett = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (7, 1), padding=(3, 0), bias=True, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        if in_channels != out_channels or stride != 1:
            if use_spatial_att:
                self.downs1 = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=True),
                    nn.BatchNorm2d(out_channels),
                )
            self.downs2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if use_temporal_att:
                self.downt1 = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 1, 1, bias=True),
                    nn.BatchNorm2d(out_channels),
                )
            self.downt2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (kernel_size, 1), (stride, 1), padding=(pad, 0), bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            if use_spatial_att:
                self.downs1 = lambda x: x
            self.downs2 = lambda x: x
            if use_temporal_att:
                self.downt1 = lambda x: x
            self.downt2 = lambda x: x

        self.soft = nn.Softmax(-2)
        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(attentiondrop)

    def forward(self, x):

        N, C, T, V = x.size()
        if self.use_spatial_att:
            attention = self.atts
            if self.use_pes:
                y = self.pes(x)
            else:
                y = x
            if self.att_s:
                q, k = torch.chunk(self.in_nets(y).view(N, 2 * self.num_subset, self.inter_channels, T, V), 2,
                                   dim=1)  # nctv -> n num_subset c'tv
                attention = attention + self.tan(
                    torch.einsum('nsctu,nsctv->nsuv', [q, k]) / (self.inter_channels * T)) * self.alphas
            if self.glo_reg_s:
                attention = attention + self.attention0s.repeat(N, 1, 1, 1)
            attention_s = self.drop(attention)
            y = torch.einsum('nctu,nsuv->nsctv', [x, attention_s]).contiguous() \
                .view(N, self.num_subset * self.in_channels, T, V)
            y = self.out_nets(y)  # nctv

            y = self.relu(self.downs1(x) + y)

            y = self.ff_nets(y)

            y = self.relu(self.downs2(x) + y)
        else:
            y = self.out_nets(x)
            y = self.relu(self.downs2(x) + y)

        if self.use_temporal_att:
            attention = self.attt
            if self.use_pet:
                z = self.pet(y)
            else:
                z = y
            if self.att_t:
                q, k = torch.chunk(self.in_nett(z).view(N, 2 * self.num_subset, self.inter_channels, T, V), 2,
                                   dim=1)  # nctv -> n num_subset c'tv
                attention = attention + self.tan(
                    torch.einsum('nsctv,nscqv->nstq', [q, k]) / (self.inter_channels * V)) * self.alphat
            if self.glo_reg_t:
                attention = attention + self.attention0t.repeat(N, 1, 1, 1)
            attention_t = self.drop(attention)
            z = torch.einsum('nctv,nstq->nscqv', [y, attention_t]).contiguous() \
                .view(N, self.num_subset * self.out_channels, T, V)
            z = self.out_nett(z)  # nctv

            z = self.relu(self.downt1(y) + z)

            z = self.ff_nett(z)

            z = self.relu(self.downt2(y) + z)
        else:
            z = self.out_nett(y)
            z = self.relu(self.downt2(y) + z)

        return z

body_group = [
    np.array([5, 6, 7, 8, 22, 23]) - 1,     # left_arm
    np.array([9, 10, 11, 12, 24, 25]) - 1,  # right_arm
    np.array([13, 14, 15, 16]) - 1,         # left_leg
    np.array([17, 18, 19, 20]) - 1,         # right_leg
    np.array([1, 2, 3, 4, 21]) - 1          # torso
]

part_group = [
    np.array([1, 2, 21]) - 1, #spine
    np.array([3, 4]) - 1, #head 
    np.array([5, 6, 7]) - 1, #left arm
    np.array([8, 22, 23]) - 1, #left hand
    np.array([9, 10, 11]) - 1, #right arm
    np.array([12, 24, 25]) - 1, #right hand
    np.array([13, 14]) - 1, #left leg
    np.array([15, 16]) - 1, #left foot
    np.array([17, 18]) - 1, #right leg
    np.array([19, 20]) - 1  #right foot
]

def get_corr_joints(parts):
    num_joints = max([max(part) for part in parts]) + 1
    res = []
    for i in range(num_joints):
        for j in range(len(parts)):
            if i in parts[j]:
                res.append(j)
                break
    return torch.Tensor(res).long()

def get_corr_parts():
    res = [4, 4, 0, 0, 1, 1, 2, 2, 3, 3]
    return torch.Tensor(res).long()

class attention_jpb(nn.Module):
    def __init__(self, in_channels, num_subset=3):
        super(attention_jpb, self).__init__()
        self.num_subset = num_subset
        self.in_nets = nn.Conv2d(in_channels, 3 * in_channels, 1, bias=True)
        self.ff_nets = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(in_channels),
        )
        
        self.relu = nn.LeakyReLU(0.1)
        self.tan = nn.Tanh()
        self.down = lambda x: x
        
        self.joints_part = get_corr_joints(part_group)
        self.joints_body = get_corr_joints(body_group)

    def forward(self, x, p_att, b_att):
        N, C, T, V = x.size()
        mid_dim = C // self.num_subset
        y = x
        q, k, v = torch.chunk(self.in_nets(y).view(N, 3 * self.num_subset, mid_dim, T, V), 3,
                           dim=1)  # nctv -> n num_subset c'tv
        attention = self.tan(torch.einsum('nsctu,nsctv->nsuv', [q, k]) / (mid_dim * T))
        
        p1_att = []
        for i in range(V):
            p0_att = []
            for j in range(V):
                p0_att.append(p_att[:,:,self.joints_part[i],self.joints_part[j]])
            p1_att.append(torch.stack(p0_att, dim=-1))
        pp_att = torch.stack(p1_att, dim=-2)
        
        b1_att = []
        for i in range(V):
            b0_att = []
            for j in range(V):
                b0_att.append(b_att[:,:,self.joints_body[i],self.joints_body[j]])
            b1_att.append(torch.stack(b0_att, dim=-1))
        bb_att = torch.stack(b1_att, dim=-2)
        
        attention = attention + pp_att + bb_att
        
        y = torch.einsum('nsuv,nsctv->nsctu', [attention, v]).contiguous().view(N, C, T, V)
        y = self.ff_nets(y)
        y = self.relu(self.down(x) + y)
        
        return y

class attention_pb(nn.Module):
    def __init__(self, in_channels, num_subset=3):
        super(attention_pb, self).__init__()
        self.num_subset = num_subset
        self.in_nets = nn.Conv2d(in_channels, 3 * in_channels, 1, bias=True)
        self.ff_nets = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(in_channels),
        )
        
        self.relu = nn.LeakyReLU(0.1)
        self.tan = nn.Tanh()
        self.down = lambda x: x
        
        self.part_body = get_corr_parts()

    def forward(self, x, b_att):
        N, C, T, V = x.size()
        mid_dim = C // self.num_subset
        y = x
        q, k, v = torch.chunk(self.in_nets(y).view(N, 3 * self.num_subset, mid_dim, T, V), 3,
                           dim=1)  # nctv -> n num_subset c'tv
        attention0 = self.tan(torch.einsum('nsctu,nsctv->nsuv', [q, k]) / (mid_dim * T))
        
        b1_att = []
        for i in range(V):
            b0_att = []
            for j in range(V):
                b0_att.append(b_att[:,:,self.part_body[i],self.part_body[j]])
            b1_att.append(torch.stack(b0_att, dim=-1))
        bb_att = torch.stack(b1_att, dim=-2)
        
        attention = attention0 + bb_att
        
        y = torch.einsum('nsuv,nsctv->nsctu', [attention, v]).contiguous().view(N, C, T, V)
        y = self.ff_nets(y)
        y = self.relu(self.down(x) + y)
        
        return y, attention0

class attention_b(nn.Module):
    def __init__(self, in_channels, num_subset=3):
        super(attention_b, self).__init__()
        self.num_subset = num_subset
        self.in_nets = nn.Conv2d(in_channels, 3 * in_channels, 1, bias=True)
        self.ff_nets = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(in_channels),
        )
        
        self.relu = nn.LeakyReLU(0.1)
        self.tan = nn.Tanh()
        self.down = lambda x: x

    def forward(self, x):
        N, C, T, V = x.size()
        mid_dim = C // self.num_subset
        y = x
        q, k, v = torch.chunk(self.in_nets(y).view(N, 3 * self.num_subset, mid_dim, T, V), 3,
                           dim=1)  # nctv -> n num_subset c'tv
        attention = self.tan(torch.einsum('nsctu,nsctv->nsuv', [q, k]) / (mid_dim * T))
        y = torch.einsum('nsuv,nsctv->nsctu', [attention, v]).contiguous().view(N, C, T, V)
        y = self.ff_nets(y)
        y = self.relu(self.down(x) + y)
        
        return y, attention

def divide_s(x, parts):
    B, C, T, V = x.size()
    part_skeleton = [torch.Tensor(part).long() for part in parts]
    
    x_sum = None
    for pa in part_skeleton:
        xx = None
        for p in pa:
            if xx is None:
                xx = x[:, :, :, p].unsqueeze(-1)
                #print(xx.size())
            else:
                xx = torch.cat((xx, x[:, :, :, p].unsqueeze(-1)), dim=-1)
        xx = xx.mean(-1) # B, C, T, M -> B, C, T   [M:the number of joint in this part]
        #print(xx.size())
        # pa_len = len(pa)
        # max_len = len(part_skeleton[0])
        # if pa_len < max_len:
        #     for _ in range(pa_len, max_len):
        #         xx = torch.cat((xx, torch.zeros_like(x[:, :, :, 0].unsqueeze(-1))), dim=-1)
        if x_sum is None:
            x_sum = xx.unsqueeze(-1)
            #print(x_sum.size())
        else:
            x_sum = torch.cat((x_sum, xx.unsqueeze(-1)), dim=-1)
    # x_sum (N, C, T, P)  [P:the number of part(5)]
    # assert x_sum.size() == (B, C, T, 5), "part_spatial divide error"
    return x_sum

class DSTANet(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_frame=32, num_subset=3, dropout=0., config=None, num_person=2,
                 num_channel=3, glo_reg_s=True, att_s=True, glo_reg_t=False, att_t=True,
                 use_temporal_att=True, use_spatial_att=True, attentiondrop=0, dropout2d=0, use_pet=True, use_pes=True):
        super(DSTANet, self).__init__()

        self.out_channels = config[-1][1]
        in_channels = config[0][0]
        
        self.input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )

        param = {
            'num_node': num_point,
            'num_subset': num_subset,
            'glo_reg_s': glo_reg_s,
            'att_s': att_s,
            'glo_reg_t': glo_reg_t,
            'att_t': att_t,
            'use_spatial_att': use_spatial_att,
            'use_temporal_att': use_temporal_att,
            'use_pet': use_pet,
            'use_pes': use_pes,
            'attentiondrop': attentiondrop
        }
        
        # self.weg1 = nn.Linear(num_point, num_point) #T V V A - T A,  V T T A - V A,  T A (V A)' - T V   [A=V]
        # self.weg2 = nn.Linear(num_frame, num_point)
        self.tan = nn.Tanh()
        
        self.graph_layers = nn.ModuleList()
        for index, (in_channels, out_channels, inter_channels, stride) in enumerate(config):
            self.graph_layers.append(
                STAttentionBlock(in_channels, out_channels, inter_channels, stride=stride, num_frame=num_frame,
                                 **param))
            num_frame = int(num_frame / stride + 0.5)
        
        self.fc = nn.Linear(self.out_channels, num_class)
        self.mlp = nn.Sequential(
            nn.Linear(self.out_channels, self.out_channels),
            nn.ReLU(),
            nn.Linear(self.out_channels, self.out_channels),
        )
        
        #intra- and inter- attention
        # self.intra_satt = spatial_intra_attention(self.out_channels, num_subset)
        self.att_b = attention_b(self.out_channels, num_subset=num_subset)
        self.att_pb = attention_pb(self.out_channels, num_subset=num_subset)
        self.att_jpb = attention_jpb(self.out_channels, num_subset=num_subset)
        # self.inter_satt = spatial_inter_attention(self.out_channels, num_subset)
        # self.inter_tatt = temporal_inter_attention(self.out_channels, num_subset=num_subset, stride=stride)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
            elif isinstance(m, nn.Linear):
                fc_init(m)
        
    def forward(self, xj, xm, xju, xmu):
        """

        :param x1: joint
        :param x2: motion
        :return: 
        """
        N, C, T, V, M = xj.shape
        Nu, _, _, _, _ = xju.shape
        x = torch.cat((xj, xm, xju, xmu), dim=0) #2(N+Nu), C, T, V, M
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = x.view(2*(N+Nu)*M, C, T, V)
        x = self.input_map(x)
        
        for i, m in enumerate(self.graph_layers):
            x = m(x)
        
        yj, ym, yju, ymu = torch.split(x, [N*M, N*M, Nu*M, Nu*M], 0) #N1, C1, T1, V1
        
        yjp = divide_s(yju, part_group)
        ymp = divide_s(ymu, part_group)
        _, Cp, Tp, Vp = yjp.shape
        assert Vp == 10
        
        yjb = divide_s(yju, body_group)
        ymb = divide_s(ymu, body_group)
        _, Cb, Tb, Vb = yjb.shape
        assert Vb == 5
        
        #intra-spatial-temporal attention
        outjb, attjb = self.att_b(yjb) #joint_body
        outmb, attmb = self.att_b(ymb) #motion_body
        outjp, attjp = self.att_pb(yjp, 0.2*attjb) #2/10
        outmp, attmp = self.att_pb(ymp, 0.2*attmb)
        outj = self.att_jpb(yju, 0.12*attjp, 0.24*attjb) #3/25  6/25
        outm = self.att_jpb(ymu, 0.12*attmp, 0.24*attmb)
        
        
        _, C1, T1, V1 = outj.shape
        _, C1p, T1p, V1p = outjp.shape
        _, C1b, T1b, V1b = outjb.shape
        outj = outj.view(Nu, M, C1, T1, V1).mean(1).mean(-1).mean(-1) #(N, C1)
        outm = outm.view(Nu, M, C1, T1, V1).mean(1).mean(-1).mean(-1)
        outjp = outjp.view(Nu, M, C1p, T1p, V1p).mean(1).mean(-1).mean(-1) #(N, C1)
        outmp = outmp.view(Nu, M, C1p, T1p, V1p).mean(1).mean(-1).mean(-1)
        outjb = outjb.view(Nu, M, C1b, T1b, V1b).mean(1).mean(-1).mean(-1) #(N, C1)
        outmb = outmb.view(Nu, M, C1b, T1b, V1b).mean(1).mean(-1).mean(-1)
        
        _, C0, T0, V0 = yj.shape
        yj = yj.view(N, M, C0, -1).permute(0, 1, 3, 2).contiguous().view(N, -1, C0, 1)  # whole channels of one spatial
        yj = yj.mean(3).mean(1)
        ym = ym.view(N, M, C0, -1).permute(0, 1, 3, 2).contiguous().view(N, -1, C0, 1)  # whole channels of one spatial
        ym = ym.mean(3).mean(1)
        
        return outj, outm, outjp, outmp, outjb, outmb, yj, ym


if __name__ == '__main__':
    config = [[64, 64, 16, 1], [64, 64, 16, 1],
              [64, 128, 32, 2], [128, 128, 32, 1],
              [128, 256, 64, 2], [256, 256, 64, 1],
              [256, 256, 64, 1], [256, 256, 64, 1],
              ]
    net = DSTANet(config=config)  # .cuda()
    ske = torch.rand([2, 3, 32, 25, 2])  # .cuda()
    print(net(ske).shape)
