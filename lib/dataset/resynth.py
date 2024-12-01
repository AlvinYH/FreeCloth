import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import numpy as np

from os.path import join
import glob
import hydra
import json
import pickle
import smplx
import trimesh

from pytorch3d.structures import Meshes
from lib.utils.sample import sample_dual_points, sample_points_with_bary

    
class ReSynthDataSet(Dataset):
    def __init__(self, params, subject_id, split):
        self.split = split
        self.params = params
        
        # load SMPL-related assets
        assets_path = hydra.utils.to_absolute_path(params.assets_path)

        # load face look-up index list for each part
        face_seg_filepath = join(assets_path, 'smplx_face_part.json')
        self.face_part_idx = json.load(open(face_seg_filepath))
        
        # load barycentric coordinates and face-id on UV map
        bary = np.load(join(assets_path, 'bary_coords_smplx_uv256.npy'))
        face_id = np.load(join(assets_path, 'uv_mask256_with_faceid_smplx.npy')).reshape(-1)
        valid_mask = (face_id >= 0).reshape(-1)
        bary = bary[valid_mask]
        face_id = face_id[valid_mask]

        # load smpl faces for sampling
        smpl_model = smplx.create(
            model_path=hydra.utils.to_absolute_path(params.smpl_model_path), 
            model_type=params.smpl_model_type,
        )
        self.smpl_faces = torch.from_numpy(smpl_model.faces.astype(np.int32)).long()
        del smpl_model

        # NOTE: only support single-subject training now
        data_path = hydra.utils.to_absolute_path(params.data_path)
        print("Load data from: ", data_path)
        print("Subject ID: ", subject_id)

        # load clothing-cut map
        with open(join(data_path, 'cut_map.pkl'), 'rb') as f:
            cut_map = pickle.load(f)[subject_id]
        
        # load different regions according to the cut map
        self.bary, self.face_id = {}, {}
        part_type_list = ['default', 'deformed', 'generated']
        for part_idx, part_type in enumerate(part_type_list):
            part_mask = (cut_map == part_idx)
            self.bary[part_type] = torch.from_numpy(bary[part_mask]).float()
            self.face_id[part_type] = torch.from_numpy(face_id[part_mask]).long()
            print("Load {} points on {} region".format(self.bary[part_type].shape[0], part_type))

        # load face mask (0: default, 1: deformed, 2: generated)
        with open(join(data_path, 'faces_mask.pkl'), 'rb') as f:
            self.smpl_faces_mask = pickle.load(f)[subject_id]
        self.no_loose_mask = (self.smpl_faces_mask < 2)  # exclude the loose part

        # load canonical template for each subject
        template_path = join(data_path, 'minimal_body_shape', subject_id+'.obj')
        mesh = trimesh.load(template_path)
        self.cano_verts = torch.from_numpy(mesh.vertices).float()
        
        # load posed data
        print("Data spacing: {}".format(params.spacing))
        if split == 'train':
            self.regstr_list = sorted(glob.glob(join(data_path, subject_id, split, subject_id+'*.npz')))[::params.spacing]
        else:
            self.regstr_list = sorted(glob.glob(join(data_path, subject_id, split, subject_id+'*.npz')))
        print("Load {} {}ing examples of subject {}".format(len(self.regstr_list), split, subject_id))
        
    def __getitem__(self, index):
        regstr = np.load(self.regstr_list[index])
        data = {}
        
        # load posed SMPLX vertices
        body_verts = torch.tensor(regstr['body_verts']).float()
        
        # resample on uv map (more uniform points); only for testing
        if self.params.uv_resample and self.split == 'test':
            # only sample from default or deformed region
            posed_pc, posed_nml, cano_pc, cano_nml, bary, masked_face_id = sample_dual_points(
                body_verts, 
                self.cano_verts, 
                self.smpl_faces[self.no_loose_mask],
                num_sample_surf=self.params.num_resample_points,
            )

            # compute the corresponding canonical points
            face_id = np.arange(self.smpl_faces.shape[0])[self.no_loose_mask][masked_face_id]
            face_type = self.smpl_faces_mask[face_id]

            # set default mask to zero out the residual
            default_mask = (face_type == 0)

        else:
            # sampling on different parts (3 types: masked, deformed, generated)
            # NOTE: we predict normal for default region
            bary = torch.cat([self.bary["default"], self.bary["deformed"]], dim=0)
            face_id = torch.cat([self.face_id["default"], self.face_id["deformed"]], dim=0)
            
            # sample points on the posed body and canonical body
            meshes = Meshes(body_verts[None], self.smpl_faces[None])
            posed_pc, posed_nml = sample_points_with_bary(meshes, bary, face_id)
            
            meshes_cano = Meshes(self.cano_verts[None], self.smpl_faces[None])
            cano_pc, cano_nml = sample_points_with_bary(meshes_cano, bary, face_id)

            default_mask = np.zeros(bary.shape[0], dtype=bool)
            default_mask[:self.bary["default"].shape[0]] = 1

        body_pc = torch.cat([posed_pc, cano_pc, cano_nml], dim=-1)
        tris = self.smpl_faces[face_id]
        
        # directly compute transformation matrix for sampled points
        vtransf = torch.tensor(regstr['vtransf']).float()
        vtransf_by_tris = vtransf[tris] 
        transf_mtx_pts = torch.einsum('pijk, pi->pjk', vtransf_by_tris, bary) 
        
        data.update({
            'body_verts': body_verts,
            'cano_verts': self.cano_verts,
            'transf': transf_mtx_pts,
            'body_pc': body_pc,
            'body_bary': bary,
            'body_tris': tris,
            'default_mask': default_mask,
        })

        # sampling on the generated region (4 different local parts)
        # NOTE: hard-coded for now
        gen_part = {'leftupleg': 2048, 'leftleg': 2048, 'rightupleg': 2048, 'rightleg': 2048}
        gen_cond_pc = []
        for (part, num_samples) in gen_part.items():
            assert part in self.face_part_idx.keys(), "Invalid face part name"
            selected_faces = self.smpl_faces[self.face_part_idx[part]]
            posed_pc, posed_nml, cano_pc, _, bary, face_id = sample_dual_points(
                body_verts, 
                self.cano_verts,
                selected_faces, 
                num_sample_surf=num_samples
            )
            
            # use cano pc for grouping in pointnet2
            part_pc = torch.cat([cano_pc, posed_pc, posed_nml], dim=-1)
            gen_cond_pc.append(part_pc)
    
        gen_cond_pc = torch.stack(gen_cond_pc, dim=0)
        
        # load gen_cond_pc and GT
        data.update({
            'gen_cond_pc': gen_cond_pc,
            'clothed_pc': torch.tensor(regstr['scan_pc']).float(),
            'clothed_nml': torch.tensor(regstr['scan_n']).float()
        })
         
        return data
    
    def __len__(self):
        return len(self.regstr_list)


class ReSynthDataModule(pl.LightningDataModule):
    def __init__(self, name, subject_id, params):
        super().__init__()
        self.name = name
        self.subject_id = subject_id
        self.params = params

    def setup(self, stage=None):
        if stage == 'fit':
            self.dataset_train = ReSynthDataSet(params=self.params, subject_id=self.subject_id, split='train')                        
        self.dataset_val = ReSynthDataSet(params=self.params, subject_id=self.subject_id, split='test')   

    def train_dataloader(self):
        dataloader = DataLoader(self.dataset_train,
                                batch_size=self.params.batch_size,
                                num_workers=self.params.num_workers, 
                                shuffle=True,
                                drop_last=True,
                                pin_memory=True)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(self.dataset_val,
                                batch_size=self.params.batch_size,
                                num_workers=self.params.num_workers, 
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True)
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(self.dataset_val,
                                batch_size=1,
                                num_workers=self.params.num_workers, 
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True)
        return dataloader