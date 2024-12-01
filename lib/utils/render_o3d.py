import numpy as np
import copy


try:
    import open3d as o3d

    # create visualizer
    color_mode = 'normal_colored'
    img_res = 1024
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=img_res, height=img_res)

    # load camera extrinsic
    EXTRINSIC = np.load('./assets/cam_front_extrinsic.npy')

except:
    print('Open3D is not installed. Please install it by running: pip install open3d')


# export point cloud to ply file
def export_pc(points, normals, ply_path):
    points = points.detach().cpu().numpy()
    normals = normals.detach().cpu().numpy()
    v_c = vertex_normal_2_vertex_color(normals)
    customized_export_pc(ply_path, points, v_n=normals, v_c=v_c)

def customized_export_pc(outfile_name, v, f = None, v_n = None, v_c = None, f_c = None, e = None):
    '''
    Author: Jinlong Yang, jyang@tue.mpg.de

    Exports a point cloud / mesh to a .ply file
    supports vertex normal and color export
    such that the saved file will be correctly displayed in MeshLab

    # v: Vertex position, N_v x 3 float numpy array
    # f: Face, N_f x 3 int numpy array
    # v_n: Vertex normal, N_v x 3 float numpy array
    # v_c: Vertex color, N_v x (3 or 4) uchar numpy array
    # f_n: Face normal, N_f x 3 float numpy array
    # f_c: Face color, N_f x (3 or 4) uchar numpy array
    # e: Edge, N_e x 2 int numpy array
    # mode: ascii or binary ply file. Value is {'ascii', 'binary'}
    '''

    v_n_flag=False
    v_c_flag=False
    f_c_flag=False

    N_v = v.shape[0]
    assert(v.shape[1] == 3)
    if not type(v_n) == type(None):
        assert(v_n.shape[0] == N_v)
        v_n_flag = True
    if not type(v_c) == type(None):
        assert(v_c.shape[0] == N_v)
        v_c_flag = True
        if v_c.shape[1] == 3:
            # warnings.warn("Vertex color does not provide alpha channel, use default alpha = 255")
            alpha_channel = np.zeros((N_v, 1), dtype = np.ubyte)+255
            v_c = np.hstack((v_c, alpha_channel))

    N_f = 0
    if not type(f) == type(None):
        N_f = f.shape[0]
        assert(f.shape[1] == 3)
        if not type(f_c) == type(None):
            assert(f_c.shape[0] == f.shape[0])
            f_c_flag = True
            if f_c.shape[1] == 3:
                # warnings.warn("Face color does not provide alpha channel, use default alpha = 255")
                alpha_channel = np.zeros((N_f, 1), dtype = np.ubyte)+255
                f_c = np.hstack((f_c, alpha_channel))
    N_e = 0
    if not type(e) == type(None):
        N_e = e.shape[0]

    with open(outfile_name, 'w') as file:
        # Header
        file.write('ply\n')
        file.write('format ascii 1.0\n')
        file.write('element vertex %d\n'%(N_v))
        file.write('property float x\n')
        file.write('property float y\n')
        file.write('property float z\n')

        if v_n_flag:
            file.write('property float nx\n')
            file.write('property float ny\n')
            file.write('property float nz\n')
        if v_c_flag:
            file.write('property uchar red\n')
            file.write('property uchar green\n')
            file.write('property uchar blue\n')
            file.write('property uchar alpha\n')

        file.write('element face %d\n'%(N_f))
        file.write('property list uchar int vertex_indices\n')
        if f_c_flag:
            file.write('property uchar red\n')
            file.write('property uchar green\n')
            file.write('property uchar blue\n')
            file.write('property uchar alpha\n')

        if not N_e == 0:
            file.write('element edge %d\n'%(N_e))
            file.write('property int vertex1\n')
            file.write('property int vertex2\n')

        file.write('end_header\n')

        # Main body:
        # Vertex
        if v_n_flag and v_c_flag:
            for i in range(0, N_v):
                file.write('%f %f %f %f %f %f %d %d %d %d\n'%\
                    (v[i,0], v[i,1], v[i,2],\
                    v_n[i,0], v_n[i,1], v_n[i,2], \
                    v_c[i,0], v_c[i,1], v_c[i,2], v_c[i,3]))
        elif v_n_flag:
            for i in range(0, N_v):
                file.write('%f %f %f %f %f %f\n'%\
                    (v[i,0], v[i,1], v[i,2],\
                    v_n[i,0], v_n[i,1], v_n[i,2]))
        elif v_c_flag:
            for i in range(0, N_v):
                file.write('%f %f %f %d %d %d %d\n'%\
                    (v[i,0], v[i,1], v[i,2],\
                    v_c[i,0], v_c[i,1], v_c[i,2], v_c[i,3]))
        else:
            for i in range(0, N_v):
                file.write('%f %f %f\n'%\
                    (v[i,0], v[i,1], v[i,2]))
        # Face
        if f_c_flag:
            for i in range(0, N_f):
                file.write('3 %d %d %d %d %d %d %d\n'%\
                    (f[i,0], f[i,1], f[i,2],\
                    f_c[i,0], f_c[i,1], f_c[i,2], f_c[i,3]))
        else:
            for i in range(0, N_f):
                file.write('3 %d %d %d\n'%\
                    (f[i,0], f[i,1], f[i,2]))

        # Edge
        if not N_e == 0:
            for i in range(0, N_e):
                file.write('%d %d\n'%(e[i,0], e[i,1]))

def vertex_normal_2_vertex_color(vertex_normal):
    # Normalize vertex normal
    normal_length = ((vertex_normal**2).sum(1))**0.5
    normal_length = normal_length.reshape(-1, 1)
    vertex_normal /= normal_length
    # Convert normal to color:
    color = vertex_normal * 255/2.0 + 128
    return color.astype(np.ubyte)


# set up camera for rendering
def set_up_camera(dist=1200):
    focal_length = dist * (img_res / 1024.) # 1200 is a hand-set focal length when the img resolution=1024. 
    x0, y0 = img_res / 2.0 - 0.5, img_res / 2.0 - 0.5
    INTRINSIC = np.array([
        [focal_length, 0.,           x0], 
        [0.,           focal_length, y0],
        [0.,           0.,            1]
    ])
    cam_intrinsics = o3d.camera.PinholeCameraIntrinsic()
    cam_intrinsics.intrinsic_matrix = INTRINSIC
    cam_intrinsics.width = img_res
    cam_intrinsics.height = img_res

    cam_params_front = o3d.camera.PinholeCameraParameters()
    cam_params_front.intrinsic = cam_intrinsics
    return cam_params_front

def render_o3d(ply_path, pt_size=5, view='front'):
    if view == 'front':
        R = np.eye(3)
    elif view == 'left':
        R = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    elif view == 'right':
        R = np.array([[0, 0, -1], [0, 1, 0], [-1, 0, 0]])
    elif view == 'back':
        R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    else:
        raise NotImplementedError
    
    # add translation along y axis 
    NEW_EXTRINSIC = copy.deepcopy(EXTRINSIC)
    NEW_EXTRINSIC[:3, :3] = R.dot(NEW_EXTRINSIC[:3, :3])
    
    cam_params_front = set_up_camera()
    cam_params_front.extrinsic = NEW_EXTRINSIC

    # read from saved file
    pcl = o3d.io.read_point_cloud(ply_path)

    vis.add_geometry(pcl)
    opt = vis.get_render_option()
    opt.point_size = pt_size
    
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(cam_params_front, allow_arbitrary=True)

    vis.poll_events()
    vis.update_renderer()

    img = vis.capture_screen_float_buffer(True)
    img = np.asarray(img)
    vis.clear_geometries()
    return img
    # rotate in xz plane
    R = np.array([[np.cos(azimuth), 0, np.sin(azimuth)], [0, 1, 0], [-np.sin(azimuth), 0, np.cos(azimuth)]])

    NEW_EXTRINSIC = copy.deepcopy(EXTRINSIC)
    NEW_EXTRINSIC[:3, :3] = R.dot(NEW_EXTRINSIC[:3, :3])

    if zoom:
        transl = np.array([0, -0.5, 0])
        NEW_EXTRINSIC[:3, 3] = NEW_EXTRINSIC[:3, 3] + transl
        cam_params_front = set_up_camera(dist=2600)
    else:
        cam_params_front = set_up_camera()
    cam_params_front.extrinsic = NEW_EXTRINSIC

    # read from saved file
    pcl = o3d.io.read_point_cloud(ply_path)

    vis.add_geometry(pcl)
    opt = vis.get_render_option()
    opt.point_size = pt_size
    
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(cam_params_front, allow_arbitrary=True)

    vis.poll_events()
    vis.update_renderer()

    img = vis.capture_screen_float_buffer(True)

    # convert to numpy array
    img = np.asarray(img)

    # vis.capture_screen_image(img_save_path, True)
    vis.clear_geometries()
    return img