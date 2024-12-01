import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG


class ResnetBlockFC(nn.Module):
    """ Fully connected ResNet Block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class ResnetPointnet(nn.Module):
    """ PointNet-based encoder network with ResNet blocks.
    Args:
        out_dim (int): dimension of latent code c
        hidden_dim (int): hidden dimension of the network
        dim (int): input dimensionality (default 3)
    """

    def __init__(self, out_dim, hidden_dim, dim=3, **kwargs):
        super().__init__()
        self.out_dim = out_dim

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.block_0 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.use_block2 = kwargs.get('use_block2', False)
        self.block_2 = ResnetBlockFC(2 * hidden_dim, hidden_dim) if self.use_block2 else None
        self.block_3 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, out_dim)

        self.act = nn.ReLU()

    @staticmethod
    def pool(x, dim=-1, keepdim=False):
        return x.max(dim=dim, keepdim=keepdim)[0]

    def forward(self, p):
        # output size: B x T X F
        p = p.permute(0, 2, 1)
        net = self.fc_pos(p)
        net = self.block_0(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        if self.use_block2:
            net = self.block_2(net)
            pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
            net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_4(net)

        # to  B x F
        net = self.pool(net, dim=1)

        c = self.fc_c(self.act(net))

        return c


class PointNet2(nn.Module):
    def __init__(self, dim=6, use_xyz=True):
        super().__init__()   
        self.use_xyz = use_xyz
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=2048,
                radii=[0.1,],
                nsamples=[32,],
                mlps=[[dim, 32, 32, 64]],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.2,],
                nsamples=[32,],
                mlps=[[64, 64, 64, 128]],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.4,],
                nsamples=[32,],
                mlps=[[128, 128, 128, 256]],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=32,
                radii=[0.8],
                nsamples=[32],
                mlps=[[256, 256, 256, 512]],
                use_xyz=True,
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + dim, 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256]))


    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz = pointcloud[..., 0:3].contiguous()
        if self.use_xyz:
            features = pointcloud.transpose(1, 2).contiguous()
        else:
            features = pointcloud[..., 3:].transpose(1, 2).contiguous()

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )
        
        return l_features[0]