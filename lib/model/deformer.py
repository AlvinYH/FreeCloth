'''
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
'''

import torch
from torch import nn

from lib.model.pointnet import ResnetPointnet, PointNet2
from lib.model.shape_decoder import ShapeDecoder


class Deformer(nn.Module):
    
    def __init__(self, dim_pose_code, dim_outfit_code, 
                 encoder_arch='pointnet', decoder_arch='shape_decoder'):
        
        super().__init__()
        
        self.dim_outfit_code = dim_outfit_code
        self.decoder_arch = decoder_arch
        
        # pose code, local code, global code, canonical xyz, posed xyz
        in_size = dim_pose_code+2*dim_outfit_code+6  
            
        if encoder_arch == 'pointnet':
            self.encoder = ResnetPointnet(dim=6, out_dim=128, hidden_dim=128)
        elif encoder_arch == 'pointnet2':
            self.encoder = PointNet2(dim=6)
        else:
            raise NotImplementedError
        
            
        if decoder_arch == 'shape_decoder':
            self.decoder = ShapeDecoder(in_size=in_size, hsize=256)
            
        elif decoder_arch == 'implicit_net':
            self.query_encoder = ImplicitNet(
                d_in=in_size,  
                d_out=128,
                dims=[256, 256, 256+in_size, 256, 256, 256],
                skip_in=[3],
                geometric_init=False,
            )
            self.def_decoder = ImplicitNet(
                d_in=6+self.query_encoder.d_out,
                d_out=3,
                dims=[256, 256+6+self.query_encoder.d_out, 256],
                skip_in=[2],
                geometric_init=False,
            )
            self.nml_decoder = ImplicitNet(
                d_in=6+self.query_encoder.d_out,
                d_out=3,
                dims=[256, 256+6+self.query_encoder.d_out, 256],
                skip_in=[2],
                geometric_init=False,
            )
        
        else:
            raise NotImplementedError
        
    def transform_points(self, p):
        pi = np.pi
        L = self.n_freq_posenc
        p_transformed = torch.cat(
            [torch.cat(
                [torch.sin((2 ** i) * pi * p),
                 torch.cos((2 ** i) * pi * p)],
                dim=-1) for i in range(L)], dim=-1)
        return p_transformed
    
    def encode(self, points):
        """
        Args:
            points: canonical points and posed points [B, N, 6]
        """
        # encode point clouds to extract local features
        latent_code = self.encoder(points).permute(0, 2, 1)
        return latent_code

    def decode(self, points, latent_code, global_outfit_code, local_outfit_code=None):
        """
        Args:
            points (torch.tensor): Input point cloud [B, N, 3+3]
            latent_code: [B, N, f_len]
            outfit_code: [B, C]
        """
    
        z_code = torch.cat((
            latent_code, 
            points[..., 3:],  # canonical points as positional encoding
            local_outfit_code,
            global_outfit_code.unsqueeze(1).expand(-1, points.shape[1], -1),  # (B, N, C)
        ), dim=-1).permute(0, 2, 1)  # [B, f_len+3+C, N]
         
        # query deformation and normals
        if self.decoder_arch == 'shape_decoder':
            deformation, normals = self.decoder(z_code) 
            deformation = deformation.permute(0, 2, 1)
            normals = normals.permute(0, 2, 1)
        
        elif self.decoder_arch == 'implicit_net':
            z_code = z_code.permute(0, 2, 1)
            query = self.query_encoder(z_code)
            deformation = self.def_decoder(torch.cat((z_code[..., :6], query), dim=-1))
            normals = self.nml_decoder(torch.cat((z_code[..., :6], query), dim=-1))

        return deformation, normals