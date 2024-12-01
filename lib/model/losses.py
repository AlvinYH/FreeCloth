import torch
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_points
from chamferdist.chamferdist import ChamferDistance
from kaolin.ops.mesh import check_sign
from kaolin.metrics.trianglemesh import point_to_mesh_distance


# use kaolin to compute sdf here!
def cal_sdf_batch(verts, faces, points):
    triangles = face_vertices(verts, faces)
    batch_size = points.shape[0]
    residues, _, _ = point_to_mesh_distance(points.contiguous(), triangles)
    # add a small epsilon to avoid nan
    pts_dist = torch.sqrt(residues.contiguous() + 1e-8)
    # pts_signs = -2.0 * (check_sign(verts.cuda(), faces[0].cuda(), points.cuda()).float() - 0.5).to(device) # negative outside
    # pts_signs = -2.0 * (check_sign(verts.cuda(), faces.cuda(), points.cuda()).float() - 0.5).to(device) # negative outside
    pts_signs = -2.0 * (check_sign(verts, faces, points).float() - 0.5) #.to(device)
    pts_sdf = (pts_dist * pts_signs).unsqueeze(-1)
    return pts_sdf.view(batch_size, -1, 1)


def face_vertices(vertices, faces):
    """ 
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    bs, nv = vertices.shape[:2]
    _, nf = faces.shape[:2]
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(vertices.device) * nv)[:, None, None]
    vertices = vertices.reshape((bs*nv, vertices.shape[-1]))
    return vertices[faces.long()]


def repulsion_loss(points, r=0.1, h=0.03, n_neighbors=20):
    B, N, _ = points.shape
    assert points.shape[-1] == 3
   
    _, nn_idx, _ = knn_points(points, points, K=n_neighbors+1, return_sorted=True)
    nn_idx = nn_idx[:, :, 1:] # exclude the first one, which is the point itself
    nn_idx = nn_idx.reshape(B, N*n_neighbors)
    
    nn_points = torch.gather(points, 1, nn_idx.unsqueeze(-1).expand(-1, -1, 3))
    nn_points = nn_points.reshape(B, N, n_neighbors, 3)

    # compute repulsion loss
    dist = torch.norm(nn_points - points.unsqueeze(2), dim=-1)  # [batch, N, n_neighbors]
    dist = torch.clamp(dist, max=r)
    weight = torch.exp(-(dist / h) **2)
    loss_repulsion = -torch.mean(dist*weight)
    return loss_repulsion


# adapted from POP
def chamfer_dist(output, target):
    cdist = ChamferDistance()
    assert output.is_contiguous() and target.is_contiguous()  # otherwise the cdist will be buggy
    model2scan, scan2model, idx1, idx2 = cdist(output, target)
    return model2scan, scan2model, idx1, idx2


def normal_loss(output_normals, target_normals, nearest_idx):
    '''
    Given the set of nearest neighbors found by chamfer distance, calculate the
    L1 discrepancy between the predicted and GT normals on each nearest neighbor point pairs.
    Note: the input normals are already normalized (length==1).
    '''
    nearest_idx = nearest_idx.expand(3, -1, -1).permute([1,2,0]).long() # [batch, N] --> [batch, N, 3], repeat for the last dim
    target_normals_chosen = torch.gather(target_normals, dim=1, index=nearest_idx)
    
    assert output_normals.shape == target_normals_chosen.shape

    # avg over the last axis
    lnormal = F.l1_loss(output_normals, target_normals_chosen, reduction='none').mean(-1)
    return lnormal, target_normals_chosen


# compute chamfer distance and normal loss
def compute_cd_nml(output, output_normals, target, target_normals):
    '''
    Given the output and target point clouds, compute the chamfer distance and normal loss.
    '''
    # compute chamfer distance
    _, s2m, idx_closest_gt, _ = chamfer_dist(output, target) #idx1: [#pred points]
    s2m = s2m.mean(1)
    
    # compute normal loss
    loss_nml, closest_target_normals = normal_loss(output_normals, target_normals, idx_closest_gt)
    nearest_idx = idx_closest_gt.expand(3, -1, -1).permute([1,2,0]).long() # [batch, N] --> [batch, N, 3], repeat for the last dim
    target_points_chosen = torch.gather(target, dim=1, index=nearest_idx)
    
    # following POP, compute the projection of m2s on the normal of the closest target points
    pc_diff = target_points_chosen - output # vectors from prediction to its closest point in gt pcl
    m2s = torch.sum(pc_diff * closest_target_normals, dim=-1) # project on direction of the normal of these gt points
    m2s = torch.mean(m2s**2, 1)
    
    return m2s, s2m, loss_nml