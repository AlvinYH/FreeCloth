import torch
import numpy as np
import pytorch_lightning as pl

import smplx
import hydra
import wandb
import os
import imageio

from lib.model.deformer import Deformer
from lib.model.sparenet import SpareNetGenerator
from lib.model.losses import *
from lib.utils.sample import fps
from lib.utils.render_o3d import export_pc, render_o3d


class FreeClothModel(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.model
        
        # NOTE: only support single-subject modeling now
        # learnable global outfit code for generation
        global_outfit_code = torch.ones(1, self.cfg.deformer.dim_outfit_code).normal_(mean=0., std=0.01).cuda()
        global_outfit_code.requires_grad = True
        self.global_outfit_code = torch.nn.Parameter(global_outfit_code)
        
        # learnable local outfit code for deformation
        local_outfit_code = torch.ones(1, cfg.smpl.n_verts, self.cfg.deformer.dim_outfit_code).normal_(mean=0., std=0.01).cuda()
        local_outfit_code.requires_grad = True
        self.local_outfit_code = torch.nn.Parameter(local_outfit_code)
        
        # LBS-based deformer
        self.deformer = Deformer(**self.cfg.deformer)

        # free-form generator      
        print("Number of primitives of generator: ", self.cfg.generator.n_primitives)
        print("Number of generated points: ", self.cfg.generator.num_points)
        print("Whether to use fps downsampling: ", self.cfg.fps_downsample)
        
        self.generator = SpareNetGenerator(**self.cfg.generator)

        # load SMPL faces (for calculating SDF)
        if self.cfg.use_collision:
            smpl_model = smplx.create(
                model_path=hydra.utils.to_absolute_path(cfg.smpl.model_path), 
                model_type=cfg.smpl.model_type
            )
            faces = torch.tensor(smpl_model.faces.astype(np.int32)).long()
            self.smpl_faces = faces.cuda()
            del smpl_model
            
        self.step_outputs = []

        # for visualization
        if self.cfg.vis.enabled:
            os.makedirs(self.cfg.vis.output_dir, exist_ok=True)
            os.makedirs(self.cfg.vis.pc_dir, exist_ok=True)


    def configure_optimizers(self):
        params_list = [
            {'params': list(self.deformer.parameters()), 'lr':self.cfg.optim.lr},
            {'params': list(self.generator.parameters()), 'lr':self.cfg.optim.lr_gen},
            {'params': self.local_outfit_code, 'lr':self.cfg.optim.lr_code},
            {'params': self.global_outfit_code, 'lr':self.cfg.optim.lr_code},
        ]
        optimizer = torch.optim.Adam(params_list)
        return optimizer
        

    def get_outfit_code(self, batch_size):
        # NOTE: only support single-subject modeling now
        batch_local_outfit_code = self.local_outfit_code.repeat(batch_size, 1, 1)
        batch_global_outfit_code = self.global_outfit_code.repeat(batch_size, 1)
        return batch_local_outfit_code, batch_global_outfit_code


    def forward(self, data):

        ### step 1: encoded features from SMPLX vertices in the canonical space ###
        cano_verts, body_verts = data['cano_verts'], data['body_verts']
        pose_code = self.deformer.encode(torch.cat([cano_verts, body_verts], dim=-1))

        ### step 2: compute interpolated pose features ###
        body_pc = data['body_pc']
        bary, tris = data['body_bary'], data['body_tris'] # [B, N, 3]

        B, N = body_pc.shape[:2]
        tris = tris.view(B, -1)
        smpl_code = torch.gather(pose_code, dim=1, 
                                 index=tris.unsqueeze(-1).repeat(1, 1, pose_code.shape[-1]))
        smpl_code = smpl_code.view(B, N, 3, pose_code.shape[-1])
        
        # barycentric interpolation
        fused_code = torch.sum(smpl_code * bary.unsqueeze(-1), dim=2)

        ### step 3: compute outfit features and predict LBS-based deformation ###
        local_outfit_code, global_outfit_code = self.get_outfit_code(B)
        smpl_outfit_code = torch.gather(local_outfit_code, dim=1, 
                                        index=tris.unsqueeze(-1).repeat(1, 1, local_outfit_code.shape[-1]))
        smpl_outfit_code = smpl_outfit_code.view(B, N, 3, local_outfit_code.shape[-1])
        fused_outfit_code = torch.sum(smpl_outfit_code * bary.unsqueeze(-1), dim=2)
        residual, deformed_nml = self.deformer.decode(body_pc, fused_code, global_outfit_code, fused_outfit_code)

        # NOTE: we still predict normal for the default part (otherwise the normal is not smooth)
        residual[data['default_mask']] = 0.0  # set residual to zero for default part

        # local to global coordinate transformation
        # compute deformed point cloud and normal
        transf = data['transf']
        residual = torch.einsum('bnij, bnj->bni', transf, residual)
        residual = residual * self.cfg.residual_scaling
        deformed_pc = body_pc[..., :3] + residual  
        deformed_nml = torch.einsum('bnij, bnj->bni', transf, deformed_nml)
        deformed_nml = torch.nn.functional.normalize(deformed_nml, dim=-1)
        
        ### step 4: free-form loose clothing generation ###
        gen_cond_pc = data['gen_cond_pc']   # [cano_pc, posed_pc, posed_nml]

        # random horizontal flip for augmentation
        if self.training and self.cfg.pose_augmentation:
            # transformation type: 1 for identity, -1 for reflection
            flip = torch.randint(2, (B,), device=gen_cond_pc.device) * 2 - 1
            flip = torch.stack([flip, torch.ones_like(flip), torch.ones_like(flip)], dim=1).unsqueeze(1)

            # apply flip to posed point cloud and normal
            gen_cond_pc[..., 3:6] = gen_cond_pc[..., 3:6] * flip.unsqueeze(1)
            gen_cond_pc[..., 6:9] = gen_cond_pc[..., 6:9] * flip.unsqueeze(1)

            # free-form generation
            gen_pc, gen_nml = self.generator(gen_cond_pc, global_outfit_code)
            
            # inverse horizontal flip
            gen_pc = gen_pc * flip
            gen_nml = gen_nml * flip
                        
        else:
            gen_pc, gen_nml = self.generator(gen_cond_pc, global_outfit_code)

        if self.training:
            assert gen_pc.shape[1] == self.cfg.generator.num_points, "Number of generated points does not match!"
            assert gen_nml.shape[1] == self.cfg.generator.num_points, "Number of generated normals does not match!"


        ### step 5: gather all points and apply FPS (optional) ###
        pred_pc = torch.cat([deformed_pc, gen_pc], dim=1)
        pred_nml = torch.cat([deformed_nml, gen_nml], dim=1)
        
        if not self.training:  # apply FPS for evaluation
            if self.cfg.fps_downsample:
                pred_pc, pred_nml = fps(pred_pc, pred_nml, self.cfg.n_all_points)
            return pred_pc, pred_nml
        
        ### step 6: compute loss ###
        # ground truth data for training supervision
        gt_pc = data['clothed_pc'].requires_grad_(True)
        gt_nml = data['clothed_nml'].requires_grad_(True)
        
        # compute loss
        m2s, s2m, loss_nml = compute_cd_nml(pred_pc, pred_nml, gt_pc, gt_nml)
        m2s, s2m = m2s.mean(), s2m.mean()
        loss_cd = m2s + s2m
        
        if self.cfg.use_normal_bidirectional:  # compute bi-directional normal loss!
            _, _, loss_nml_new = compute_cd_nml(gt_pc, gt_nml, pred_pc, pred_nml)
            loss_nml = (loss_nml.mean() + loss_nml_new.mean()) / 2
        else:
            loss_nml = loss_nml.mean()

        # regularize the deformation
        loss_reg = torch.mean(residual ** 2)   

        # regularization on the latent code
        loss_latent_global = torch.mean(global_outfit_code**2)
        loss_latent_local = torch.mean(local_outfit_code**2)
        
        loss_dict = {
            'cd': loss_cd,
            'nml': loss_nml,
            'reg': loss_reg,
            'latent_global': loss_latent_global,
            'latent_local': loss_latent_local,
        }

        if self.cfg.use_collision:  # collision loss
            # only consider the penetration loss of the generated points
            sdf = cal_sdf_batch(data['body_verts'], self.smpl_faces, gen_pc)
            sdf[sdf > -self.cfg.collision_threshold] = 0  # set a soft threshold

            # penalize all negative sdf
            loss_collision = torch.nn.functional.relu(-sdf).sum() / sdf.shape[0]        
            loss_dict["collision"] = loss_collision

        if self.cfg.use_repulsion:   # repulsion loss; not used
            # enforce the generated points not to intersect with the deformed point
            loss_repulsion = repulsion_loss(pred_pc, r=self.cfg.radius_repulsion)
            loss_dict["repulsion"] = loss_repulsion
        
        return pred_pc, pred_nml, loss_dict
    

    def training_step(self, data, data_idx):
        pred_pc, pred_nml, loss_dict = self.forward(data)
        
        loss = 0
        for k, v in loss_dict.items():
            assert "lambda_" + k in self.cfg, "Loss weight of {:s} is not defined!".format(k)
            if k == 'nml':   # add normal loss only after cd loss is stable
                if self.current_epoch < self.cfg.epoch_start_nml:
                    loss += 1e-6 * v  # very small weight
                    continue
            loss += eval('self.cfg.lambda_' + k) * v
               
        self.log('loss', loss)
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)
        
        # visualization during training
        if self.cfg.vis.enabled and self.global_step % self.cfg.vis.n_every_train == 0:
                
            pc_dir = os.path.join(self.cfg.vis.pc_dir, 'train')
            os.makedirs(pc_dir, exist_ok=True)
                
            # save predicted point cloud
            pred_path = '{}/{:03d}.ply'.format(pc_dir, data_idx)
            export_pc(pred_pc[0], pred_nml[0], pred_path)

            # save the deformed / generated part separately
            def_path = '{}/{:03d}_def.ply'.format(pc_dir, data_idx)
            gen_path = '{}/{:03d}_gen.ply'.format(pc_dir, data_idx)
            export_pc(pred_pc[0, :-self.cfg.generator.num_points], pred_nml[0, :-self.cfg.generator.num_points], def_path)
            export_pc(pred_pc[0, -self.cfg.generator.num_points:], pred_nml[0, -self.cfg.generator.num_points:], gen_path)

            if self.cfg.vis.render_o3d:
                img_gen = render_o3d(gen_path)
                img_def = render_o3d(def_path)
                img_pred = render_o3d(pred_path, view='front')

                # save gt point cloud and render
                gt_path = '{}/{:03d}_gt.ply'.format(pc_dir, data_idx)
                if not os.path.exists(gt_path):
                    export_pc(data['clothed_pc'][0], data['clothed_nml'][0], gt_path)
                img_gt = render_o3d(gt_path)

                img = np.concatenate([img_gen, img_def, img_pred, img_gt], axis=1)
                wandb.log({"image": [wandb.Image(img, caption=f'train_step_{self.global_step}')]})

        return loss
    

    def validation_step(self, data, data_idx):
        with torch.no_grad():
            pred_pc, pred_nml = self.forward(data)
            
            # compute CD and NML
            m2s, s2m, loss_nml = compute_cd_nml(pred_pc, pred_nml, data['clothed_pc'], data['clothed_nml'])
            loss_cd = m2s + s2m
            
            if self.cfg.vis.enabled and data_idx == 0:  # visualize the first frame for validation
                pc_dir = os.path.join(self.cfg.vis.pc_dir, 'valid')
                os.makedirs(pc_dir, exist_ok=True)
                
                # save predicted point cloud
                pred_path = '{}/{:03d}.ply'.format(pc_dir, data_idx)
                export_pc(pred_pc[0], pred_nml[0], pred_path)

                if self.cfg.vis.render_o3d:
                    num_views = 4   # render 4 views
                    name_views = ['front', 'back', 'left', 'right']
                    img_pred = []
                    for k in range(num_views):
                        img_pred.append(render_o3d(pred_path, view=name_views[k]))
                    img_pred = np.concatenate(img_pred, axis=1)
        
                    # save gt point cloud and render
                    gt_path = '{}/{:03d}_gt.ply'.format(pc_dir, data_idx)
                    if not os.path.exists(gt_path):
                        export_pc(data['clothed_pc'][0], data['clothed_nml'][0], gt_path)
                    img_gt = render_o3d(gt_path)

                    img = np.concatenate([img_pred, img_gt], axis=1)
                    wandb.log({"image": [wandb.Image(img, caption=f'valid_step_{self.global_step}')]})
 
        output = {'cd':loss_cd, 'nml': loss_nml, 'm2s': m2s, 's2m': s2m}
        self.step_outputs.append(output)
        return


    def on_validation_epoch_end(self, test=False):
        loss_cd, loss_nml  = [], []
        m2s, s2m = [], []
        for output in self.step_outputs:
            loss_cd.append(output['cd'])
            loss_nml.append(output['nml'])
            m2s.append(output['m2s'])
            s2m.append(output['s2m'])

        # compute average metric
        loss_cd = torch.cat(loss_cd, dim=0)
        loss_nml = torch.cat(loss_nml, dim=0).mean(-1)
        m2s = torch.cat(m2s, dim=0)
        s2m = torch.cat(s2m, dim=0)

        # save evaluated metric to txt file
        if not test:
            with open('metric_valid.txt', 'a') as f:
                f.write("Epoch {:d}, CD: {:.3e}, NML: {:.3e}, M2S: {:.3e}, S2M: {:.3e}\n".format(
                    self.current_epoch, loss_cd.mean(), loss_nml.mean(), m2s.mean(), s2m.mean()
                ))
        else:
            with open('metric_test.txt', 'a') as f:
                f.write("Epoch {:d}, CD: {:.3e}, NML: {:.3e}, M2S: {:.3e}, S2M: {:.3e}\n".format(
                    self.current_epoch, loss_cd.mean(), loss_nml.mean(), m2s.mean(), s2m.mean()
                ))
        
        self.log('valid_model2scan', m2s.mean())
        self.log('valid_scan2model', s2m.mean())
        self.log('valid_cd', loss_cd.mean())
        self.log('valid_nml', loss_nml.mean())
        print("CD: {:.3e}, NML: {:.3e}".format(loss_cd.mean(), loss_nml.mean()))

        self.step_outputs.clear()  # free memory


    def test_step(self, data, data_idx): 
        with torch.no_grad():
            pred_pc, pred_nml = self.forward(data)
            
            # evaluation
            m2s, s2m, loss_nml = compute_cd_nml(pred_pc, pred_nml, data['clothed_pc'], data['clothed_nml'])
            loss_cd = m2s + s2m

            # visualization
            if data_idx % self.cfg.vis.n_every_test == 0:
                output_dir = os.path.join(self.cfg.vis.output_dir, 'test')
                pc_dir = os.path.join(self.cfg.vis.pc_dir, 'test')
                
                os.makedirs(output_dir, exist_ok=True)
                os.makedirs(pc_dir, exist_ok=True)

                # save predicted point cloud
                pred_path = '{}/{:03d}.ply'.format(pc_dir, data_idx)
                export_pc(pred_pc[0], pred_nml[0], pred_path)

                if self.cfg.vis.render_o3d:
                    num_views = 4   # render from 4 views
                    name_views = ['front', 'back', 'left', 'right']
                    img_pred = []
                    for k in range(num_views):
                        img_pred.append(render_o3d(pred_path, view=name_views[k]))
                    img_pred = np.concatenate(img_pred, axis=1)
        
                    # save gt point cloud and render
                    gt_path = '{}/{:03d}_gt.ply'.format(pc_dir, data_idx)
                    if not os.path.exists(gt_path):
                        export_pc(data['clothed_pc'][0], data['clothed_nml'][0], gt_path)
                    img_gt = render_o3d(gt_path)

                    img = np.concatenate([img_pred, img_gt], axis=1)
                    img = (img * 255).astype(np.uint8)
                    imageio.imwrite(os.path.join(output_dir, '{:03d}.png'.format(data_idx)), img)

        output = {'cd':loss_cd, 'nml': loss_nml, 'm2s': m2s, 's2m': s2m}
        self.step_outputs.append(output)
        return
            
            
    def on_test_epoch_end(self):
        return self.on_validation_epoch_end(test=True)