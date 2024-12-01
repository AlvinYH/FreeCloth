import torch
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation
from typing import Tuple
import sys

from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
from pytorch3d.ops.packed_to_padded import packed_to_padded
from pytorch3d.structures import Meshes


# generate random barycentric coordinates
def _rand_barycentric_coords(
    size1, size2, dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Helper function to generate random barycentric coordinates which are uniformly
    distributed over a triangle.

    Args:
        size1, size2: The number of coordinates generated will be size1*size2.
                      Output tensors will each be of shape (size1, size2).
        dtype: Datatype to generate.
        device: A torch.device object on which the outputs will be allocated.

    Returns:
        w0, w1, w2: Tensors of shape (size1, size2) giving random barycentric
            coordinates
    """
    uv = torch.rand(2, size1, size2, dtype=dtype, device=device)
    u, v = uv[0], uv[1]
    u_sqrt = u.sqrt()
    w0 = 1.0 - u_sqrt
    w1 = u_sqrt * (1.0 - v)
    w2 = u_sqrt * v
    return w0, w1, w2


# sample points from mesh with given barycentric coordinates
def sample_points_with_bary(meshes, bary, sample_face_idxs):
    verts = meshes.verts_packed()
    faces = meshes.faces_packed()

    # Get the vertex coordinates of the sampled faces.
    face_verts = verts[faces] # (F, 3, 3)
    sample_face_verts = face_verts[sample_face_idxs] # (num_samples, 3, 3)
    
    # Use the barycentric coords to get a point on each sampled face.
    samples = torch.sum(sample_face_verts * bary[:, :, None], dim=1)
    
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]
    vert_normals = (v1 - v0).cross(v2 - v1, dim=1)
    vert_normals = vert_normals / vert_normals.norm(dim=1, p=2, keepdim=True).clamp(
        min=sys.float_info.epsilon
    )
    normals = vert_normals[sample_face_idxs]
    return samples, normals


def sample_dual_points_from_meshes(meshes, meshes_cano, num_samples=20000):
    verts = meshes.verts_packed()
    faces = meshes.faces_packed()
    mesh_to_face = meshes.mesh_to_faces_packed_first_idx()
    num_meshes = len(meshes)

    with torch.no_grad():
        areas, _ = mesh_face_areas_normals(verts, faces)  # Face areas can be zero.
        max_faces = meshes.num_faces_per_mesh().max().item()
        areas_padded = packed_to_padded(areas, mesh_to_face, max_faces)  # (N, F)

        # (gkioxari) Confirm multinomial bug is not present with real data.
        sample_face_idxs = areas_padded.multinomial(num_samples, replacement=True)  # (N, num_samples)
        sample_face_idxs += mesh_to_face.view(num_meshes, 1)

    # Get the vertex coordinates of the sampled faces.
    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Randomly generate barycentric coords.
    w0, w1, w2 = _rand_barycentric_coords(
        num_meshes, num_samples, verts.dtype, verts.device
    )

    # Use the barycentric coords to get a point on each sampled face.
    a = v0[sample_face_idxs]  # (N, num_samples, 3)
    b = v1[sample_face_idxs]
    c = v2[sample_face_idxs]
    samples = w0[:, :, None] * a + w1[:, :, None] * b + w2[:, :, None] * c
    bary = torch.stack([w0, w1, w2], dim=-1)
    
    vert_normals = (v1 - v0).cross(v2 - v1, dim=1)
    vert_normals = vert_normals / vert_normals.norm(dim=1, p=2, keepdim=True).clamp(
        min=sys.float_info.epsilon
    )
    normals = vert_normals[sample_face_idxs]
    
    # NOTE: compute corresponding points from canonical mesh
    # no need to sample again
    verts_cano = meshes_cano.verts_packed()
    faces_cano = meshes_cano.faces_packed()
    face_verts_cano = verts_cano[faces_cano]
    v0, v1, v2 = face_verts_cano[:, 0], face_verts_cano[:, 1], face_verts_cano[:, 2]
    a = v0[sample_face_idxs]  # (N, num_samples, 3)
    b = v1[sample_face_idxs]
    c = v2[sample_face_idxs]
    samples_cano = w0[:, :, None] * a + w1[:, :, None] * b + w2[:, :, None] * c
    
    vert_normals_cano = (v1 - v0).cross(v2 - v1, dim=1)
    vert_normals_cano = vert_normals_cano / vert_normals_cano.norm(dim=1, p=2, keepdim=True).clamp(
        min=sys.float_info.epsilon
    )
    normals_cano = vert_normals_cano[sample_face_idxs]
    
    return samples, normals, samples_cano, normals_cano, bary, sample_face_idxs


# sample points from dual space (posed and canonical) 
def sample_dual_points(posed_verts, cano_verts, faces, num_sample_surf):
    mesh = Meshes(posed_verts[None], faces[None])
    mesh_cano = Meshes(cano_verts[None], faces[None])
    pc, nml, pc_cano, nml_cano, bary, face_idxs = sample_dual_points_from_meshes(mesh, mesh_cano, num_sample_surf)
    # NOTE: squeeze the batch dimension
    return pc[0], nml[0], pc_cano[0], nml_cano[0], bary[0], face_idxs[0]


# farthest point sampling (with normals)
def fps(xyz, normals, num_points):
    xyz_flipped = xyz.transpose(1, 2).contiguous()
    normals_flipped = normals.transpose(1, 2).contiguous()
    sample_idxs = furthest_point_sample(xyz, num_points)
    new_xyz = gather_operation(xyz_flipped, sample_idxs).transpose(1, 2).contiguous()
    new_normals = gather_operation(normals_flipped, sample_idxs).transpose(1, 2).contiguous()
    return new_xyz, new_normals