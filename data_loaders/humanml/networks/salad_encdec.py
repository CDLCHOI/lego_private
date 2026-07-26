import torch
import torch.nn as nn

from .salad_conv import ResSTConv, get_activation
from .salad_pool import STPool, STUnpool, adj_list_to_edges

kit_adj_list = [
    [1, 11, 16],
    [0, 2],
    [1, 3],
    [2, 4, 5, 8],
    [3],
    [3, 6],
    [5, 7],
    [6],
    [3, 9],
    [8, 10],
    [9],
    [0, 12],
    [11, 13],
    [12, 14],
    [13, 15],
    [14],
    [0, 17],
    [16, 18],
    [17, 19],
    [18, 20],
    [19],
]
t2m_adj_list = [
    [1, 2, 3],
    [0, 4],
    [0, 5],
    [0, 6],
    [1, 7],
    [2, 8],
    [3, 9],
    [4, 10],
    [5, 11],
    [6, 12, 13, 14],
    [7],
    [8],
    [9, 15],
    [9, 16],
    [9, 17],
    [12],
    [13, 18],
    [14, 19],
    [16, 20],
    [17, 21],
    [18],
    [19],
]

class MotionEncoderSALAD(nn.Module):
    def __init__(self, opt) -> None:
        super(MotionEncoderSALAD, self).__init__()
        self.motion_enc = MotionEncoder(opt)
        self.conv_enc = STConvEncoder(opt)
        self.linear = nn.Linear(7*opt.latent_dim, opt.dim_movement_latent) # 7*32, 512

    def forward(self, x):
        x = self.motion_enc(x) # (b,196,263) -> (b,196,22,32)
        x = self.conv_enc(x) # (b,49,7,32)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.linear(x)
        return x # (b,49,512)


class MotionEncoder(nn.Module):
    def __init__(self, opt):
        super(MotionEncoder, self).__init__()

        self.dim_pose = opt.dim_pose
        self.joints_num = (self.dim_pose + 1) // 12
        self.latent_dim = opt.latent_dim
        self.contact_joints = opt.contact_joints

        self.layers = nn.ModuleList()
        for i in range(self.joints_num):
            if i == 0:
                input_dim = 7
            elif i in self.contact_joints:
                input_dim = 13
            else:
                input_dim = 12
            self.layers.append(nn.Sequential(
                nn.Linear(input_dim, self.latent_dim),
                get_activation(opt.activation),
                nn.Linear(self.latent_dim, self.latent_dim),
            ))

    def forward(self, x):
        """
        x: [bs, nframes, dim_pose]
        
        nfeats = 12J + 1
            - root_rot_velocity (B, seq_len, 1)
            - root_linear_velocity (B, seq_len, 2)
            - root_y (B, seq_len, 1)
            - ric_data (B, seq_len, (joint_num - 1)*3)
            - rot_data (B, seq_len, (joint_num - 1)*6)
            - local_velocity (B, seq_len, joint_num*3)
            - foot contact (B, seq_len, 4)
        """
        B, T, D = x.size()

        # split
        root, ric, rot, vel, contact = torch.split(x, [4, 3 * (self.joints_num - 1), 6 * (self.joints_num - 1), 3 * self.joints_num, 4], dim=-1)
        ric = ric.reshape(B, T, self.joints_num - 1, 3)
        rot = rot.reshape(B, T, self.joints_num - 1, 6)
        vel = vel.reshape(B, T, self.joints_num, 3)

        # joint-wise input
        joints = [torch.cat([root, vel[:, :, 0]], dim=-1)] # [B, T, 7]]
        for i in range(1, self.joints_num):
            joints.append(torch.cat([ric[:, :, i - 1], rot[:, :, i - 1], vel[:, :, i]], dim=-1))
        for cidx, jidx in enumerate(self.contact_joints):
            joints[jidx] = torch.cat([joints[jidx], contact[:, :, cidx, None]], dim=-1)
        # 这里len(joints)=22，都是(32,196,d)，维度，根7，4个脚步节点是12+1，其他都是12=3+6+3
        # encode
        out = []
        for i in range(self.joints_num):
            out.append(self.layers[i](joints[i]))
        out = torch.stack(out, dim=2)

        return out # (b,196,22,32)


class STConvEncoder(nn.Module):
    def __init__(self, opt):
        super(STConvEncoder, self).__init__()
        self.opt = opt
        # adjacency list
        self.adj_list = {
            "t2m": t2m_adj_list,
            "kit": kit_adj_list,
        }[opt.dataset_name]

        # topology
        self.edge_list = [adj_list_to_edges(self.adj_list)]
        self.mapping_list = []

        # network
        self.layers = nn.ModuleList()
        for i in range(opt.n_layers):
            layers = []
            for _ in range(opt.n_extra_layers):
                layers.append(ResSTConv(
                    self.edge_list[-1],
                    opt.latent_dim,
                    opt.kernel_size,
                    activation=opt.activation,
                    norm=opt.norm,
                    dropout=opt.dropout
                ))
            layers.append(ResSTConv(
                self.edge_list[-1],
                opt.latent_dim,
                opt.kernel_size,
                activation=opt.activation,
                norm=opt.norm,
                dropout=opt.dropout
            ))

            pool = STPool(opt.dataset_name, i)
            layers.append(pool)
            self.layers.append(nn.Sequential(*layers))

            self.edge_list.append(pool.new_edges)
            self.mapping_list.append(pool.skeleton_mapping)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

