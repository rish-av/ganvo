from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch
import datetime
from os.path import join
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

pixel_coords = None


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def split(joint_dataset,val_percent, sampler=SubsetRandomSampler,random_seed=42):

    '''
    function useful there is no explicit valitdation scripts available
    joint_dataset = train + val items
    returns train and validation samplers based on sampling strategy
    '''
    dataset_size = len(joint_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_percent * dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    return train_sampler, valid_sampler



def get_log_dir():
    '''
    New log dir at every run according to the time at that point in time.
    '''
    now = datetime.datetime.now()
    return "logs/run-%d-%d-%d-%d-%d-%d"%(now.year,now.month,now.day,now.hour,now.minute,now.second)


def get_summary_writer(rootdir):

    return SummaryWriter(join(rootdir,get_log_dir()))


def grad(im):
    gradx = F.pad((im[:, :, :, 2:] - im[:, :, :, :-2]) / 2.0, (1, 1, 0, 0))
    grady = F.pad((im[:, :, 2:, :] - im[:, :, :-2, :]) / 2.0, (0, 0, 1, 1))
    return gradx, grady 

def sobel(im):
    c = im.size()[1]
    fx = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    fx = fx.view(1, 1, 3, 3).expand(1, c, 3, 3)
    fy = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    fy = fy.view(1, 1, 3, 3).expand(1, c, 3, 3)
    if im.is_cuda:
        fx = fx.cuda()
        fy = fy.cuda()
    gradx = F.pad(F.conv2d(im, fx), (1, 1, 1, 1))
    grady = F.pad(F.conv2d(im, fy), (1, 1, 1, 1))
    return gradx, grady

def set_id_grid(depth):
    global pixel_coords
    b, _,  h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1, h, w).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]

def euler2mat(angle):
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat


def quat2mat(quat):
    norm_quat = torch.cat([quat[:, :1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat



def pose_vec2mat(vec, rotation_mode='euler'):
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:, 3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr):
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    X_norm = 2*(X / Z)/(w-1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.reshape(b, h, w, 2)


def pixel2cam(depth, intrinsics_inv):
    global pixel_coords
    b,_,h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size(2) < h:
        set_id_grid(depth)
    current_pixel_coords = pixel_coords[..., :h, :w].expand(b, 3, h, w).reshape(b, 3, -1)  # [B, 3, H*W]
    cam_coords = (intrinsics_inv @ current_pixel_coords).reshape(b, 3, h, w)
    return cam_coords * depth



def inverse_warp(img, depth, pose, intrinsics, rotation_mode='euler', padding_mode='zeros'):

    batch_size, _, img_height, img_width = img.size()

    cam_coords = pixel2cam(depth, intrinsics.inverse())  # [B,3,H,W]
    pose_mat = pose_vec2mat(pose, rotation_mode)  # [B,3,4]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[..., :3], proj_cam_to_src_pixel[..., -1:]
    src_pixel_coords = cam2pixel(cam_coords, rot, tr)  # [B,H,W,2]
    projected_img = F.grid_sample(img, src_pixel_coords, padding_mode=padding_mode, align_corners=True)

    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1

    return projected_img, valid_points
