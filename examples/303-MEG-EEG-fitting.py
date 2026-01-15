# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import functools
import os
import pickle
from typing import Union, Callable

import braintools
import brainunit as u
import jax.tree
import matplotlib.pyplot as plt
import mne
import nibabel
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.io import loadmat
from sklearn.metrics.pairwise import cosine_similarity

import brainmass
import brainstate
from brainstate.nn import GaussianReg, Param, Const, ReluT, ExpT, ClipT

Parameter = Union[brainstate.nn.Param, brainstate.typing.ArrayLike, Callable]
Initializer = Union[Callable, brainstate.typing.ArrayLike]

brainstate.environ.set(dt=0.0001 * u.second)


def get_language_data():
    # We use an example dataset for one subject on a public Google Drive folder
    output_dir = 'D:/codes/projects/whole-brain-nmm-pytorch/data_ismail2025/'

    # We will use MEG data recorded during a covert verb generation task in verb generation trials and noise trials
    # Evoked MEG data averaged across trials (-100 to 400 ms)
    verb_meg_raw = np.load(os.path.join(output_dir, 'verb_evoked.npy'))  # (time, channels)
    noise_meg_raw = np.load(os.path.join(output_dir, 'noise_evoked.npy'))  # (time, channels)
    # Normalize both signals
    verb_meg = verb_meg_raw / np.abs(verb_meg_raw).max() * 1
    noise_meg = noise_meg_raw / np.abs(noise_meg_raw).max() * 1

    # We will use the leadfield to simulate MEG activty from sources derived from the individual's head model
    leadfield = loadmat(os.path.join(output_dir, 'leadfield_3d.mat'))  # shape (sources, sensors, 3)
    lm_3d = leadfield['M']  # 3D leadfield matrix
    # Convert 3D to 2D using SVD-based projection
    lm = np.zeros_like(lm_3d[:, :, 0])
    for sources in range(lm_3d.shape[0]):
        u_, d, v = np.linalg.svd(lm_3d[sources])
        lm[sources] = u_[:, :3].dot(np.diag(d)).dot(v[0])
    # Scale the leadfield matrix
    lm = lm.T / 1e-11 * 5  # Shape: (channels, sources)

    # We will use the individual's weights and distance matrices
    sc_df = pd.read_csv(os.path.join(output_dir, 'weights.csv'), header=None).values
    sc = np.log1p(sc_df)
    sc = sc / np.linalg.norm(sc)
    dist = np.loadtxt(os.path.join(output_dir, 'distance.txt'))
    node_size = sc.shape[0]

    # Format input data
    data_verb = verb_meg.T
    data_noise = noise_meg.T

    # To simulate the auditory inputs in this task we will stimulate the auditory cortices
    # These nodes were identified using an ROI mask of left and right Heschl's gyri
    # based on the Talairach Daemon database
    ki0 = np.zeros(node_size)
    ki0[[2, 183, 5]] = 1

    uu = np.zeros((data_verb.shape[0], node_size))
    uu[100:140] = ki0

    return lm, sc, dist, data_verb, uu


def get_hdeeg_data():
    # Select the session number to use:
    # Please do not change it as we are using subject-specific anatomy
    ses2use = 10

    # download data
    data_dir = 'D:/codes/projects/whole-brain-nmm-pytorch/data_momi2025'

    # Load Schaefer 200-parcel atlas data
    atlas200_file = os.path.join(data_dir, 'Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv')
    atlas200 = pd.read_csv(atlas200_file)

    # Load Schaefer 1000-parcel atlas data
    atlas1000_file = os.path.join(data_dir, 'Schaefer2018_1000Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv')
    atlas1000 = pd.read_csv(atlas1000_file)

    # load the structural connectivity file
    sc_file = os.path.join(data_dir, 'Schaefer2018_200Parcels_7Networks_count.csv')

    # distance file
    dist_file = os.path.join(data_dir, 'Schaefer2018_200Parcels_7Networks_distance.csv')

    # Load the precomputed EEG evoked response data from a file
    all_eeg_evoked = np.load(data_dir + '/empirical_data/all_eeg_evoked.npy')

    # Read the epoch data from an MNE-formatted file
    epo_eeg = mne.read_epochs(data_dir + '/empirical_data/example_epoched.fif', verbose=False)

    # Compute the average evoked response from the epochs
    evoked = epo_eeg.average()

    # Replace the data of the averaged evoked response with data from the selected session
    evoked.data = all_eeg_evoked[ses2use]

    # Load additional data from pickle files
    with open(data_dir + '/empirical_data/all_epo_seeg.pkl', 'rb') as handle:
        all_epo_seeg = pickle.load(handle)
    with open(data_dir + '/empirical_data/dist_Schaefer_1000parcels_7net.pkl', 'rb') as handle:
        dist_Schaefer_1000parcels_7net = pickle.load(handle)

    # Extract the stimulation region data from the loaded pickle file
    stim_region = dist_Schaefer_1000parcels_7net['stim_region']

    # Extract coordinates and ROI labels from the atlas data
    coords_200 = np.array([atlas200['R'], atlas200['A'], atlas200['S']]).T

    # Remove network names from the ROI labels for clarity
    label_stripped_200 = list(map(lambda x: x.replace('7Networks_', ''), atlas200['ROI Name']))
    # label = atlas200['ROI Name']
    # label_stripped_200 = []
    # for xx in range(len(label)):
    #     label_stripped_200.append(label[xx].replace('7Networks_', ''))

    # Extract coordinates and ROI labels from the atlas data
    coords_1000 = np.array([atlas1000['R'], atlas1000['A'], atlas1000['S']]).T
    ROI_Name = atlas1000['ROI Name']

    # Remove network names from the ROI labels for clarity
    label_stripped_1000 = list(map(lambda x: x.replace('7Networks_', ''), ROI_Name))

    # Find the index of the stimulation region in the list of stripped ROI labels (1000 parcels)
    stim_idx = label_stripped_1000.index(stim_region[ses2use])

    # Use the index to get the coordinates of the stimulation region from the 1000-parcel atlas
    stim_coords = coords_1000[stim_idx]

    # Extract the network name from the stimulation region label
    # The network name is the part after the underscore in the stimulation region label
    stim_net = stim_region[ses2use].split('_')[1]

    # python
    # vectorized Euclidean distances between coords_200 (N,3) and stim_coords (3,)
    coords_200 = np.asarray(coords_200)  # ensure array shape (N,3)
    stim_coords = np.ravel(stim_coords)  # ensure shape (3,)

    # fast, memory-friendly:
    distances = np.linalg.norm(coords_200 - stim_coords, axis=1)

    # Iterate over the indices of the distances array, sorted in ascending order
    for idx, item in enumerate(np.argsort(distances)):
        # Check if the network name of the stimulation region is present in the label of the current parcel
        if stim_net in label_stripped_200[item]:
            # If the condition is met, assign the index
            # of the current parcel to `parcel2inject`
            parcel2inject = item
            # Exit the loop since the desired parcel has been found
            break

    # pick session array, take absolute and ensure float for safe in-place ops
    keys = list(all_epo_seeg.keys())
    abs_value = np.abs(all_epo_seeg[keys[ses2use]]).astype(np.float64, copy=False)

    # subtract per-channel mean using broadcasting (axis=1 = channels/rows)
    abs_value -= abs_value.mean(axis=1, keepdims=True)

    # take absolute of the demeaned signals
    abs_value = np.abs(abs_value)

    # Find the starting and ending points around the maximum value in the data
    # Get the index of the maximum value along the time axis
    starting_point = np.where(abs_value == abs_value.max())[1][0] - 10
    ending_point = np.where(abs_value == abs_value.max())[1][0] + 10

    # Compute the maximum, mean, and standard deviation of the data within the range around the maximum
    mean = np.mean(abs_value[:, starting_point:ending_point])
    std = np.std(abs_value[:, starting_point:ending_point])

    # Define a threshold as mean + 4 times the standard deviation
    thr = mean + (4 * std)

    # Count the number of unique regions affected by the threshold
    number_of_region_affected = np.unique(np.where(abs_value > thr)[0]).shape[0]

    # Load the rewritten Schaeffer 200 parcels
    atlas200_img = nibabel.load(
        data_dir + '/calculate_distance/example_Schaefer2018_200Parcels_7Networks_rewritten.nii')

    # Get the shape and affine matrix of the image
    shape, affine = atlas200_img.shape[:3], atlas200_img.affine

    # Create a meshgrid of voxel coordinates
    coords = np.array(np.meshgrid(*(range(i) for i in shape), indexing='ij'))

    # Rearrange the coordinates array to have the correct shape
    coords = np.rollaxis(coords, 0, len(shape) + 1)

    # Apply the affine transformation to get the coordinates in millimeters
    mm_coords = nibabel.affines.apply_affine(affine, coords)

    data = np.asarray(atlas200_img.get_fdata(), dtype=np.int32)  # label image (voxel values = parcel indices)
    coords_flat = mm_coords.reshape(-1, 3)  # shape (n_voxels, 3)
    labels_flat = data.ravel()  # shape (n_voxels,)

    n_parcels = 200
    minlength = max(int(labels_flat.max()) + 1, n_parcels + 1)  # ensure we have bins up to 200

    # sum coordinates per label for each axis using bincount
    sums = np.vstack([
        np.bincount(labels_flat, weights=coords_flat[:, 0], minlength=minlength),
        np.bincount(labels_flat, weights=coords_flat[:, 1], minlength=minlength),
        np.bincount(labels_flat, weights=coords_flat[:, 2], minlength=minlength),
    ])  # shape (3, minlength)

    # voxel counts per label
    counts = np.bincount(labels_flat, minlength=minlength)  # shape (minlength,)

    # compute centroids (keep zeros for labels with no voxels to match original initialization)
    centroids = np.zeros((3, minlength), dtype=float)
    nonzero = counts > 0
    centroids[:, nonzero] = sums[:, nonzero] / counts[nonzero]

    # extract parcels 1..200 into sub_coords with shape (3, 200)
    sub_coords = centroids[:, 1:n_parcels + 1].copy()

    # Vectorized Euclidean distances between each parcel centroid and the parcel to inject
    coords = sub_coords.T.astype(float)  # shape: (n_parcels, 3)
    center = sub_coords[:, parcel2inject].astype(float)  # shape: (3,)
    distances = np.linalg.norm(coords - center, axis=1)  # shape: (n_parcels,)

    # Find the indices of the closest parcels to inject, based on the number of affected regions
    inject_stimulus = np.argsort(distances)[:number_of_region_affected]

    # Compute stimulus weights based on the distances
    # Adjust distances to a scale of 0 to 1 and calculate the values for the stimulus weights
    values = (np.max(distances[inject_stimulus] / 10) + 0.5) - (distances[inject_stimulus] / 10)

    # Initialize an array for stimulus weights with zeros
    stim_weights_thr = np.zeros((len(atlas200['ROI Name'])))
    # Assign the computed values to the stimulus weights for the selected parcels
    stim_weights_thr[inject_stimulus] = values

    # File paths for transformation, source space, and BEM files
    trans = data_dir + '/anatomical/example-trans.fif'
    src = data_dir + '/anatomical/example-src.fif'
    bem = data_dir + '/anatomical/example-bem.fif'

    # Create a forward solution using the provided transformation, source space, and BEM files
    # Only EEG is used here; MEG is disabled
    fwd = mne.make_forward_solution(
        epo_eeg.info,
        trans=trans,
        src=src,
        bem=bem,
        meg=False,
        eeg=True,
        mindist=5.0,
        n_jobs=2,
        verbose=False,
    )

    # Convert the forward solution to a fixed orientation with surface orientation
    fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True)

    # Update the leadfield matrix to use the fixed orientation
    leadfield = fwd_fixed['sol']['data']

    # Extract vertex indices for each hemisphere from the forward solution
    vertices = [src_hemi['vertno'] for src_hemi in fwd_fixed['src']]

    # Read annotation files for left and right hemispheres
    lh_vertices = nibabel.freesurfer.io.read_annot(
        data_dir + '/anatomical/lh.Schaefer2018_200Parcels_7Networks_order.annot')[0]
    rh_vertices = nibabel.freesurfer.io.read_annot(
        data_dir + '/anatomical/rh.Schaefer2018_200Parcels_7Networks_order.annot')[0]

    # Extract vertices corresponding to the parcels from the annotation files
    # Add 100 to right hemisphere vertices to adjust for parcel numbering
    lh_vertices_thr = lh_vertices[vertices[0]]
    rh_vertices_thr = rh_vertices[vertices[1]] + 100
    # Combine left and right hemisphere vertices into a single array
    vertices_thr = np.concatenate([lh_vertices_thr, rh_vertices_thr])

    # python
    labels = np.asarray(vertices_thr, dtype=np.int64)  # parcel labels per vertex
    n_parcels = 200
    minlength = n_parcels + 1  # include label 0

    # Sum leadfield values per label for each channel using np.bincount
    sums = np.vstack([
        np.bincount(labels, weights=leadfield[ch], minlength=minlength)
        for ch in range(leadfield.shape[0])
    ])  # shape: (channels, minlength)

    # Counts per label
    counts = np.bincount(labels, minlength=minlength)  # shape: (minlength,)

    # Take only parcels 1..200 and compute safe mean (avoid divide-by-zero)
    sums_sub = sums[:, 1:n_parcels + 1]  # shape: (channels, n_parcels)
    counts_sub = counts[1:n_parcels + 1]  # shape: (n_parcels,)

    with np.errstate(divide='ignore', invalid='ignore'):
        new_leadfield = sums_sub / counts_sub[np.newaxis, :]  # shape: (channels, n_parcels)

    # Set columns with zero counts to zero (no vertices for that parcel)
    zero_mask = counts_sub == 0
    if np.any(zero_mask):
        new_leadfield[:, zero_mask] = 0.0

    # Load structural connectivity data from a CSV file
    sc = pd.read_csv(sc_file, header=None, sep=' ').values
    # Apply log transformation and normalization to the structural connectivity matrix
    sc = np.log1p(sc) / np.linalg.norm(np.log1p(sc))

    # Load the distance data from the saved CSV file
    dist = pd.read_csv(dist_file, header=None, sep=' ').values

    # Extract and normalize EEG data from the evoked response
    eeg_data = evoked.data
    eeg_data = eeg_data[:, 200:600].T / (np.abs(eeg_data)).max() * 2

    # Initialize the leadfield matrix for the model
    lm = new_leadfield.copy() / 10

    # Apply stimulus at time steps 65-75
    uu = np.zeros((eeg_data.shape[0], dist.shape[0]))
    uu[65:75] = stim_weights_thr
    return lm, sc, dist, eeg_data, uu


def _get_leadfield(eeg_info):
    # 数据根目录路径
    # 包含图谱文件、连接组数据、EEG记录、解剖文件等
    data_dir = 'D:/codes/projects/whole-brain-nmm-pytorch/data_momi2025'

    # =========================================================================
    # 第十一部分：构建EEG前向模型
    # =========================================================================
    # 前向模型描述神经源活动如何投射到头皮EEG电极
    # 需要：头部几何结构、组织电导率、源空间定义

    # 定义解剖文件路径
    # trans: 头部坐标系到MRI坐标系的变换矩阵
    trans = data_dir + '/anatomical/example-trans.fif'
    # src: 源空间定义（皮层表面上的源点位置）
    src = data_dir + '/anatomical/example-src.fif'
    # bem: 边界元模型（描述头部组织层的几何结构和电导率）
    bem = data_dir + '/anatomical/example-bem.fif'

    # 计算前向解
    # 这是计算密集型操作，将源空间每个点的活动映射到EEG电极
    fwd = mne.make_forward_solution(
        eeg_info,  # EEG传感器信息（电极位置等）
        trans=trans,  # 坐标变换
        src=src,  # 源空间
        bem=bem,  # 边界元模型
        meg=False,  # 不计算MEG
        eeg=True,  # 只计算EEG
        mindist=5.0,  # 源点到内颅骨的最小距离（毫米），避免数值问题
        n_jobs=2,  # 并行计算的作业数
        verbose=False,  # 不输出详细日志
    )

    # 将前向解转换为固定方向
    # surf_ori=True: 使用皮层表面法向量作为源方向
    # force_fixed=True: 强制使用固定方向（而非自由方向）
    # use_cps=True: 使用皮层斑块统计
    fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True)

    # 提取leadfield矩阵
    # 形状为 (n_channels, n_sources)，描述每个源对每个电极的贡献
    leadfield = fwd_fixed['sol']['data']

    # 提取每个半球的源空间顶点索引
    # vertices[0]: 左半球顶点索引
    # vertices[1]: 右半球顶点索引
    vertices = [src_hemi['vertno'] for src_hemi in fwd_fixed['src']]

    # =========================================================================
    # 第十二部分：将Leadfield从源空间映射到脑区空间
    # =========================================================================
    # 目标：将高分辨率的源级leadfield转换为脑区级leadfield
    # 方法：对每个脑区内的所有源点取平均

    # 读取FreeSurfer的annotation文件，获取每个顶点所属的脑区标签
    # annotation文件是FreeSurfer格式的表面标签文件
    lh_vertices = nibabel.freesurfer.io.read_annot(
        data_dir + '/anatomical/lh.Schaefer2018_200Parcels_7Networks_order.annot')[0]
    rh_vertices = nibabel.freesurfer.io.read_annot(
        data_dir + '/anatomical/rh.Schaefer2018_200Parcels_7Networks_order.annot')[0]

    # 获取源空间顶点对应的脑区标签
    # 左半球标签保持不变（1-100）
    lh_vertices_thr = lh_vertices[vertices[0]]
    # 右半球标签加100，使其范围变为101-200
    rh_vertices_thr = rh_vertices[vertices[1]] + 100
    # 合并两半球的标签
    vertices_thr = np.concatenate([lh_vertices_thr, rh_vertices_thr])

    # 准备bincount计算
    labels = np.asarray(vertices_thr, dtype=np.int64)  # 每个源点的脑区标签
    n_parcels = 200
    minlength = n_parcels + 1  # 包含标签0（未分配的顶点）

    # 使用bincount计算每个脑区的leadfield和
    # 对每个EEG通道，按脑区标签分组求和
    sums = np.vstack([
        np.bincount(labels, weights=leadfield[ch], minlength=minlength)
        for ch in range(leadfield.shape[0])
    ])  # 形状 (n_channels, minlength)

    # 计算每个脑区包含的源点数量
    counts = np.bincount(labels, minlength=minlength)  # 形状 (minlength,)

    # 提取脑区1-200的数据（跳过标签0）
    sums_sub = sums[:, 1:n_parcels + 1]  # 形状 (n_channels, 200)
    counts_sub = counts[1:n_parcels + 1]  # 形状 (200,)

    # 计算平均值，处理可能的除零情况
    # np.errstate 临时忽略除零警告
    with np.errstate(divide='ignore', invalid='ignore'):
        new_leadfield = sums_sub / counts_sub[np.newaxis, :]  # 形状 (n_channels, 200)

    # 将没有源点的脑区（counts=0）的leadfield设为0
    zero_mask = counts_sub == 0
    if np.any(zero_mask):
        new_leadfield[:, zero_mask] = 0.0

    # 准备leadfield矩阵
    # 除以10进行缩放，调整数值范围以匹配模型的其他参数
    lm = new_leadfield.copy() / 10

    return lm


def _get_stim_weight(seeg: np.ndarray, stim_region: str):
    data_dir = 'D:/codes/projects/whole-brain-nmm-pytorch/data_momi2025'

    # =========================================================================
    # 第二部分：加载脑区图谱数据
    # =========================================================================
    # Schaefer图谱是一种基于功能连接的大脑皮层分区方案
    # 将大脑皮层划分为多个功能均匀的脑区，按7个经典静息态网络组织
    # 7个网络包括：视觉(Vis)、躯体运动(SomMot)、背侧注意(DorsAttn)、
    #             腹侧注意(SalVentAttn)、边缘(Limbic)、控制(Cont)、默认模式(Default)

    # 加载200脑区版本的图谱质心坐标
    # CSV文件包含每个脑区的名称和RAS坐标（Right-Anterior-Superior，神经影像标准坐标系）
    atlas200 = pd.read_csv(
        os.path.join(data_dir, 'Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv')
    )

    # 加载1000脑区版本的图谱质心坐标
    # 更精细的分区用于精确定位刺激位置，然后映射回200脑区模型
    atlas1000 = pd.read_csv(
        os.path.join(data_dir, 'Schaefer2018_1000Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv')
    )

    # =========================================================================
    # 第五部分：处理200脑区图谱的坐标和标签
    # =========================================================================

    # 从atlas200 DataFrame中提取RAS坐标
    # R(Right): 左右轴，正值指向右侧
    # A(Anterior): 前后轴，正值指向前方
    # S(Superior): 上下轴，正值指向上方
    # 结果形状为 (200, 3)，每行是一个脑区的三维坐标
    coords_200 = np.array([atlas200['R'], atlas200['A'], atlas200['S']]).T

    # 清理脑区标签，移除网络前缀 '7Networks_'
    # 原始标签如 '7Networks_LH_Vis_1' 变为 'LH_Vis_1'
    # 使用 map + lambda 函数进行批量字符串替换，比for循环更简洁
    label_stripped_200 = list(map(lambda x: x.replace('7Networks_', ''), atlas200['ROI Name']))

    # =========================================================================
    # 第六部分：处理1000脑区图谱的坐标和标签
    # =========================================================================

    # 提取1000脑区图谱的RAS坐标，形状为 (1000, 3)
    coords_1000 = np.array([atlas1000['R'], atlas1000['A'], atlas1000['S']]).T

    # 同样清理1000脑区的标签
    label_stripped_1000 = list(map(lambda x: x.replace('7Networks_', ''), atlas1000['ROI Name']))

    # =========================================================================
    # 第七部分：定位刺激区域并映射到200脑区空间
    # =========================================================================
    # 目标：将1000脑区图谱中的精确刺激位置映射到200脑区模型
    # 策略：找到距离最近且属于同一功能网络的200脑区

    # 在1000脑区标签列表中查找当前session刺激区域的索引
    stim_idx = label_stripped_1000.index(stim_region)

    # 获取刺激区域在1000脑区图谱中的精确坐标
    stim_coords = coords_1000[stim_idx]

    # 提取刺激区域所属的功能网络名称
    # 标签格式为 '{半球}_{网络}_{编号}'，例如 'LH_Vis_1'
    # split('_')[1] 提取第二部分，即网络名称 'Vis'
    stim_net = stim_region.split('_')[1]

    # 确保数组形状正确，为后续向量化计算做准备
    coords_200 = np.asarray(coords_200)  # 确保形状为 (200, 3)
    stim_coords = np.ravel(stim_coords)  # 展平为一维数组，形状 (3,)

    # 计算所有200个脑区到刺激点的欧氏距离
    # 使用向量化操作，比循环快得多
    # coords_200 - stim_coords 利用广播机制，得到 (200, 3) 的差值矩阵
    # np.linalg.norm(..., axis=1) 沿第二个轴计算范数，得到 (200,) 的距离向量
    distances = np.linalg.norm(coords_200 - stim_coords, axis=1)

    # 在200脑区中找到最佳匹配的脑区
    # 条件：1) 距离刺激点最近  2) 属于同一功能网络
    # np.argsort(distances) 返回按距离从小到大排序的索引
    for idx, item in enumerate(np.argsort(distances)):
        # 检查当前脑区是否属于刺激所在的网络
        if stim_net in label_stripped_200[item]:
            # 找到符合条件的脑区，记录其索引
            parcel2inject = item
            # 跳出循环，不再继续搜索
            break

    # =========================================================================
    # 第八部分：分析sEEG响应，确定受影响的脑区数量
    # =========================================================================
    # sEEG记录提供了刺激扩散的直接证据
    # 通过统计阈值方法确定有多少个脑区被电刺激显著激活

    # 获取当前session的sEEG数据
    # 取绝对值并转换为float64，确保后续数值计算的精度
    abs_value = np.abs(seeg).astype(np.float64, copy=False)

    # 去除基线：减去每个通道的均值
    # keepdims=True 保持维度，使广播机制正常工作
    # 这一步消除了不同通道之间的直流偏移差异
    abs_value -= abs_value.mean(axis=1, keepdims=True)

    # 再次取绝对值（因为去均值后可能产生负值）
    abs_value = np.abs(abs_value)

    # 找到最大响应的时间点，并定义其周围的分析窗口
    # np.where 返回满足条件的索引，[1][0] 取时间轴上的第一个最大值位置
    # 窗口为最大值前后各10个时间点
    max_idx = np.where(abs_value == abs_value.max())[1][0]
    starting_point = max_idx - 10
    ending_point = max_idx + 10

    # 在分析窗口内计算统计量
    mean = np.mean(abs_value[:, starting_point:ending_point])
    std = np.std(abs_value[:, starting_point:ending_point])

    # 定义激活阈值：均值 + 4倍标准差
    # 4倍标准差是一个相对严格的阈值，确保只有真正被激活的区域才被计入
    thr = mean + (4 * std)

    # 统计超过阈值的独立通道数量
    # np.where(abs_value > thr)[0] 获取超过阈值的通道索引
    # np.unique 去重，因为同一通道可能在多个时间点超过阈值
    number_of_region_affected = np.unique(np.where(abs_value > thr)[0]).shape[0]

    # =========================================================================
    # 第九部分：从NIfTI图像计算精确的脑区质心坐标
    # =========================================================================
    # 直接从体素级别的parcellation图像计算质心
    # 比使用atlas CSV中的预计算坐标更精确

    # 加载200脑区的NIfTI格式parcellation图像
    # 每个体素的值是其所属脑区的标签（1-200，0表示背景）
    atlas200_img = nibabel.load(
        data_dir + '/calculate_distance/example_Schaefer2018_200Parcels_7Networks_rewritten.nii')

    # 获取图像的三维形状和仿射变换矩阵
    # affine矩阵将体素坐标（整数索引）转换为毫米坐标（MNI空间）
    shape, affine = atlas200_img.shape[:3], atlas200_img.affine

    # 创建体素坐标的三维网格
    # np.meshgrid 生成坐标网格，indexing='ij' 使用矩阵索引方式
    coords = np.array(np.meshgrid(*(range(i) for i in shape), indexing='ij'))

    # 重排数组维度
    # 从 (3, x, y, z) 变为 (x, y, z, 3)，方便后续的仿射变换
    coords = np.rollaxis(coords, 0, len(shape) + 1)

    # 应用仿射变换，将体素坐标转换为毫米坐标（MNI空间）
    mm_coords = nibabel.affines.apply_affine(affine, coords)
    coords_flat = mm_coords.reshape(-1, 3)  # 展平为 (n_voxels, 3)

    # 获取parcellation标签数据并展平
    data = np.asarray(atlas200_img.get_fdata(), dtype=np.int32)  # 形状 (x, y, z)
    labels_flat = data.ravel()  # 展平为 (n_voxels,)

    # 设置bincount的参数
    n_parcels = 200
    # minlength确保结果数组至少有201个元素（0到200）
    minlength = max(int(labels_flat.max()) + 1, n_parcels + 1)

    # 使用bincount计算每个脑区的坐标和
    # bincount(labels, weights=coords) 将coords值按labels分组求和
    # 对x, y, z三个坐标轴分别计算
    sums = np.vstack([
        np.bincount(labels_flat, weights=coords_flat[:, 0], minlength=minlength),
        np.bincount(labels_flat, weights=coords_flat[:, 1], minlength=minlength),
        np.bincount(labels_flat, weights=coords_flat[:, 2], minlength=minlength),
    ])  # 形状 (3, minlength)

    # 计算每个脑区包含的体素数量
    counts = np.bincount(labels_flat, minlength=minlength)  # 形状 (minlength,)

    # 计算质心坐标 = 坐标和 / 体素数
    centroids = np.zeros((3, minlength), dtype=float)
    nonzero = counts > 0  # 找出有体素的脑区
    centroids[:, nonzero] = sums[:, nonzero] / counts[nonzero]

    # 提取脑区1到200的质心坐标（跳过标签0，即背景）
    # 形状从 (3, 200) 准备转换为 (200, 3)
    sub_coords = centroids[:, 1:n_parcels + 1].copy()

    # =========================================================================
    # 第十部分：计算刺激扩散权重
    # =========================================================================
    # 基于距离的刺激权重：距离刺激点越近的脑区，权重越大

    # 转置坐标数组，变为 (n_parcels, 3) 的形状
    coords = sub_coords.T.astype(float)
    # 获取刺激注入脑区的质心坐标
    center = sub_coords[:, parcel2inject].astype(float)
    # 计算所有脑区到刺激点的距离
    distances = np.linalg.norm(coords - center, axis=1)

    # 选择距离最近的N个脑区作为受影响区域
    # N = number_of_region_affected，由sEEG分析确定
    inject_stimulus = np.argsort(distances)[:number_of_region_affected]

    # 计算距离依赖的刺激权重
    # 公式设计确保：1) 权重为正值  2) 距离越近权重越大  3) 权重平滑衰减
    # distances/10 是缩放因子，避免数值过大
    # +0.5 确保最远的受影响区域也有正的基础权重
    values = (np.max(distances[inject_stimulus] / 10) + 0.5) - (distances[inject_stimulus] / 10)

    return values, inject_stimulus


def get_hdeeg_data_v2():
    """
    加载并预处理HD-EEG电刺激数据，用于神经质量模型拟合。

    返回值
    ------
    lm : np.ndarray
        形状 (n_channels, 200)，脑区级leadfield矩阵（已缩放）
        描述每个脑区的神经活动如何投射到头皮EEG电极

    sc : np.ndarray
        形状 (200, 200)，归一化的对数变换结构连接矩阵
        来自扩散MRI纤维追踪，表示脑区间的白质连接强度

    dist : np.ndarray
        形状 (200, 200)，脑区间欧氏距离矩阵（单位：毫米）
        用于计算神经信号的传导延迟

    eeg_data : np.ndarray
        形状 (400, n_channels)，目标EEG时间序列（已归一化）
        模型需要拟合的真实EEG诱发响应数据

    uu : np.ndarray
        形状 (400, 200)，外部刺激输入矩阵
        定义了刺激的时间窗口和空间分布
    """

    # =========================================================================
    # 第一部分：基础配置
    # =========================================================================

    # 选择要使用的实验session编号（0索引）
    # 注意：不要更改此值，因为后续使用的解剖数据是针对特定被试的
    # 如果更改session，需要同时更换对应的解剖文件（trans, src, bem等）
    ses2use = 10

    # 数据根目录路径
    # 包含图谱文件、连接组数据、EEG记录、解剖文件等
    data_dir = 'D:/codes/projects/whole-brain-nmm-pytorch/data_momi2025'

    # =========================================================================
    # 第三部分：加载EEG数据
    # =========================================================================

    # 加载所有session的预计算EEG诱发响应数据
    # 形状为 (n_sessions, n_channels, n_timepoints)
    # 诱发响应是对同一刺激多次重复后的平均EEG信号，可降低噪声
    all_eeg_evoked = np.load(data_dir + '/empirical_data/all_eeg_evoked.npy')

    # 读取MNE格式的EEG epoch数据
    # Epoch是按试次（trial）分割的EEG数据段，每个epoch对应一次刺激
    # 这里主要是为了获取EEG的元数据（电极位置、采样率等信息）
    epo_eeg = mne.read_epochs(data_dir + '/empirical_data/example_epoched.fif', verbose=False)

    # 计算epoch的平均诱发响应，创建MNE的Evoked对象
    # Evoked对象包含了传感器信息、时间轴等元数据
    evoked = epo_eeg.average()

    # 用指定session的数据替换evoked对象中的数据
    # 这样做的目的是：保留evoked对象的元数据结构，同时使用特定session的实际数据
    evoked.data = all_eeg_evoked[ses2use]

    # =========================================================================
    # 第四部分：加载sEEG数据和刺激区域信息
    # =========================================================================
    # sEEG（立体定向脑电图）是植入颅内的深部电极记录
    # 提供刺激扩散的"地面真值"，用于确定电刺激激活了多少个脑区

    # 加载所有session的sEEG epoch数据
    # 字典格式，键为session标识符，值为对应的sEEG数据数组
    with open(data_dir + '/empirical_data/all_epo_seeg.pkl', 'rb') as handle:
        all_epo_seeg = pickle.load(handle)

    # 加载包含刺激区域信息的pickle文件
    # 其中'stim_region'字段存储了每个session的刺激位置（在1000脑区图谱中的名称）
    with open(data_dir + '/empirical_data/dist_Schaefer_1000parcels_7net.pkl', 'rb') as handle:
        dist_Schaefer_1000parcels_7net = pickle.load(handle)

    # 提取刺激区域名称列表
    # 每个元素是一个字符串，格式如 'LH_Vis_1'（左半球_视觉网络_第1个脑区）
    stim_region = dist_Schaefer_1000parcels_7net['stim_region']

    # =========================================================================
    # 第十三部分：加载结构连接和距离矩阵
    # =========================================================================

    # 加载结构连接矩阵
    # 来自扩散MRI纤维追踪，值表示脑区间的白质纤维束数量
    # 定义结构连接矩阵文件路径
    # 结构连接(SC)来自扩散MRI的纤维追踪，表示脑区间的解剖连接强度
    sc = pd.read_csv(
        os.path.join(data_dir, 'Schaefer2018_200Parcels_7Networks_count.csv'),
        header=None,
        sep=' '
    ).values

    # 对结构连接矩阵进行对数变换和归一化
    # log1p(x) = log(1+x)，处理零值和压缩动态范围
    # 除以L2范数进行归一化，使不同被试的数据具有可比性
    sc = np.log1p(sc) / np.linalg.norm(np.log1p(sc))

    # 加载脑区间距离矩阵
    # 用于在神经质量模型中计算信号传导延迟

    # 定义脑区间距离矩阵文件路径
    # 距离矩阵用于计算神经信号在脑区间传播的时间延迟
    dist = pd.read_csv(
        os.path.join(data_dir, 'Schaefer2018_200Parcels_7Networks_distance.csv'),
        header=None,
        sep=' '
    ).values

    # =========================================================================
    # 第十四部分：准备最终的模型输入
    # =========================================================================

    # 提取EEG数据并进行预处理
    eeg_data = evoked.data
    # 提取时间窗口200-600（约200ms到600ms，包含主要的诱发响应）
    # .T 转置为 (time, channels) 格式
    # 归一化到 [-2, 2] 范围，便于模型训练
    eeg_data = eeg_data[:, 200:600].T / (np.abs(eeg_data)).max() * 2

    # 创建外部刺激输入矩阵
    stim_weight, indices = _get_stim_weight(
        all_epo_seeg[ses2use],
        stim_region[ses2use],
    )

    # 形状 (n_timepoints, n_regions) = (400, 200)
    uu = np.zeros((eeg_data.shape[0], dist.shape[0]))
    # 在时间步65-75应用刺激（对应刺激发生的时间窗口）
    # 刺激的空间分布由stim_weights_thr定义
    uu[65:75, indices] = stim_weight

    # 构建EEG前向模型
    lm = _get_leadfield(epo_eeg.info)

    # =========================================================================
    # 返回所有模型输入
    # =========================================================================
    # lm: leadfield矩阵，将脑区活动映射到EEG
    # sc: 结构连接矩阵，定义脑区间的连接强度
    # dist: 距离矩阵，用于计算传导延迟
    # eeg_data: 目标EEG数据，模型需要拟合的真实信号
    # uu: 外部刺激输入，定义刺激的时空模式
    return lm, sc, dist, eeg_data, uu


class BrainModelTR(brainstate.nn.Module):
    """
    Whole-brain neural mass model for EEG simulation.
    """

    def __init__(
        self,
        tr_dynamics: brainstate.nn.Dynamics,
        leadfield: brainstate.nn.Module,
        tr: u.Quantity = 0.001 * u.second,
    ):
        super().__init__()

        self.dynamics = tr_dynamics
        self.leadfield = leadfield

        # TR parameters
        self.tr = tr
        self.dt_per_tr = int(tr / brainstate.environ.get_dt())

    def update(self, tr_inputs, record_state: bool = False):
        fn_tr = lambda inp_tr: self.dynamics(inp_tr, record_state=record_state)

        if record_state:
            activities = brainstate.transform.for_loop(fn_tr, tr_inputs)
            obv = self.leadfield(activities[0])
            return obv, activities[1]
        else:
            activities = brainstate.transform.for_loop(fn_tr, tr_inputs)
            obv = self.leadfield(activities)
            return obv


class HORN_TR(brainstate.nn.Dynamics):
    def __init__(
        self,
        n_hidden,

        # structural coupling parameters
        sc: np.ndarray,
        dist: np.ndarray,
        mu: float,

        # dynamics parameters
        alpha: Parameter = 0.04,  # excitability
        omega: Parameter = 2. * u.math.pi / 28.,  # natural frequency
        gamma: Parameter = 0.01,  # damping
        v: Parameter = 0.0,  # feedback
        state_init: Callable = braintools.init.ZeroInit(),
        delay_init: Callable = braintools.init.ZeroInit(),
        tr: u.Quantity = 1e-3 * u.second,
    ):
        super().__init__(n_hidden)

        # dynamics
        dynamics = brainmass.HORNStep(
            n_hidden, alpha=alpha, omega=omega, gamma=gamma, v=v, state_init=state_init)
        delay_time = dist / mu * brainstate.environ.get_dt()
        self.h2h = brainmass.AdditiveCoupling(
            dynamics.prefetch_delay('y', delay_time, brainmass.delay_index(n_hidden), init=delay_init),
            brainmass.LaplacianConnParam(sc),
        )
        self.dynamics = dynamics
        self.tr = tr

    def update(self, inputs, record_state: bool = False):
        def step(i):
            self.dynamics(self.h2h() + inputs)

        n_step = int(self.tr / brainstate.environ.get_dt())
        brainstate.transform.for_loop(step, np.arange(n_step))
        if record_state:
            return self.dynamics.x.value, {'x': self.dynamics.x.value, 'y': self.dynamics.y.value}
        else:
            return self.dynamics.x.value


class HORN_TR2(brainstate.nn.Dynamics):
    def __init__(
        self,
        n_hidden,

        # structural coupling parameters
        sc: np.ndarray,
        dist: np.ndarray,
        mu: float,

        # dynamics parameters
        alpha: Parameter = 0.04,  # excitability
        omega: Parameter = 2. * u.math.pi / 28.,  # natural frequency
        gamma: Parameter = 0.01,  # damping
        v: Parameter = 0.0,  # feedback
        state_init: Callable = braintools.init.ZeroInit(),
        delay_init: Callable = braintools.init.ZeroInit(),
        tr: u.Quantity = 1e-3 * u.second,
    ):
        super().__init__(n_hidden)

        # dynamics
        dynamics = brainmass.HORNStep(
            n_hidden, alpha=alpha, omega=omega, gamma=gamma, v=v, state_init=state_init)
        delay_time = dist / mu * brainstate.environ.get_dt()
        self.h2h = brainmass.AdditiveCoupling(
            dynamics.prefetch_delay(
                'y', delay_time, brainmass.delay_index(n_hidden),
                init=delay_init, update_every=tr,
            ),
            brainmass.LaplacianConnParam(sc),
        )
        self.dynamics = dynamics
        self.tr = tr

    def update(self, inputs, record_state: bool = False):
        def step(i):
            self.dynamics(inp)

        inp = self.h2h() + inputs
        n_step = int(self.tr / brainstate.environ.get_dt())
        brainstate.transform.for_loop(step, np.arange(n_step))
        if record_state:
            return self.dynamics.x.value, {'x': self.dynamics.x.value, 'y': self.dynamics.y.value}
        else:
            return self.dynamics.x.value


class HORNNetworkTR(BrainModelTR):
    """
    HORN neural mass network for EEG simulation.
    """

    def __init__(
        self,
        n_hidden: int,

        # structural coupling parameters
        sc: np.ndarray,
        dist: np.ndarray,
        mu: float,

        # leadfield parameters
        cy0: Parameter,
        lm: Parameter,
        y0: Parameter,

        # other parameters
        tr: u.Quantity = 0.001 * u.second,
    ):
        # dynamics parameters
        alpha: Parameter = 0.04  # excitability
        omega: Parameter = 2. * u.math.pi / 28.  # natural frequency
        gamma: Parameter = 0.01  # damping
        v: Parameter = 0.0  # feedback

        # hyperparameters
        alpha = 0.04  # excitability
        omega_base = 2. * u.math.pi / 28.  # natural frequency
        gamma_base = 0.01  # damping
        omega_min = 0.5 * omega_base
        omega_max = 2.0 * omega_base
        gamma_min = 0.5 * gamma_base
        gamma_max = 2.0 * gamma_base
        omega = braintools.init.Uniform(omega_min, omega_max)(n_hidden)
        gamma = braintools.init.Uniform(gamma_min, gamma_max)(n_hidden)
        alpha = braintools.init.Uniform(0.005, 0.4)(n_hidden)

        # omega = Param(omega, t=SoftplusT(0.001))
        # gamma = Param(gamma, t=SoftplusT(0.001))

        # omega = Param(omega, t=ReluT(0.1))
        # gamma = Param(gamma, t=ReluT(0.1))

        omega = Param(omega, t=ClipT(0.01, 1.0))
        gamma = Param(gamma, t=ReluT(1e-4))
        alpha = Param(alpha, t=ClipT(0.005, 0.4))

        # state_init: Callable = braintools.init.ZeroInit()
        # delay_init: Callable = braintools.init.ZeroInit()

        state_init = braintools.init.Uniform(-0.01, 0.01)
        delay_init = braintools.init.Uniform(-0.01, 0.01)

        # dynamics = HORN_TR(
        dynamics = HORN_TR2(
            n_hidden,
            sc=sc,
            dist=dist,
            mu=mu,
            alpha=alpha,
            omega=omega,
            gamma=gamma,
            v=v,
            state_init=state_init,
            delay_init=delay_init,
        )

        # leadfiled matrix
        leadfield = brainmass.LeadfieldReadout(lm=lm, y0=y0, cy0=cy0)

        # super initialization
        super().__init__(dynamics, leadfield, tr=tr)


class JansenRitNetworkTR(BrainModelTR):
    """
    Jansen-Rit neural mass network for EEG simulation.
    """

    def __init__(
        self,
        node_size: int,

        # structural coupling parameters
        sc: np.ndarray,
        dist: np.ndarray,
        mu: float,

        # leadfield parameters
        cy0: Parameter,
        lm: Parameter,
        y0: Parameter,

        # other parameters
        tr: u.Quantity = 0.001 * u.second,
    ):
        g_l = Const(400)
        g_f = Const(10)
        g_b = Const(10)
        # std_in uses ExpT (no reg in original code)
        std_in = Param(6.0, t=ExpT(5.0))
        # Fixed parameters
        vmax = Const(5)
        v0 = Const(6)
        r = Const(0.56)
        # Fixed parameters
        kE = Const(0)
        kI = Const(0)
        k = Param(5.5, t=ReluT(0.5), reg=GaussianReg(5.5, 0.2, fit_hyper=True))
        # Array parameters
        lm_base = 0.01 * brainstate.random.randn_like(lm)
        lm_noise = 0.1 * brainstate.random.randn_like(lm)
        lm = Param(lm + lm_base + lm_noise)
        w_bb = Param(np.full((node_size, node_size), 0.05, dtype=brainstate.environ.dftype()))
        w_ff = Param(np.full((node_size, node_size), 0.05, dtype=brainstate.environ.dftype()))
        w_ll = Param(np.full((node_size, node_size), 0.05, dtype=brainstate.environ.dftype()))
        # initialization
        state_init = braintools.init.Uniform(-0.01, 0.01)
        delay_init = braintools.init.Uniform(-0.01, 0.01)

        dynamics = brainmass.JansenRitTR(
            in_size=node_size,

            # distance parameters
            delay=dist / mu,

            # structural parameters
            sc=sc,
            k=k,
            w_ll=w_ll,
            w_ff=w_ff,
            w_bb=w_bb,
            g_l=g_l,
            g_f=g_f,
            g_b=g_b,

            # other parameters
            state_init=state_init,
            delay_init=delay_init,
        )
        leadfield = brainmass.LeadfieldReadout(lm=lm, y0=y0, cy0=cy0)

        super().__init__(dynamics, leadfield, tr=tr)


class ModelFitting:
    def __init__(
        self,
        model: BrainModelTR,
        data: np.ndarray,
        optimizer: braintools.optim.Optimizer,
    ):
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.weights = model.states(brainstate.ParamState)
        self.optimizer.register_trainable_weights(self.weights)
        # define masks for getting lower triangle matrix indices
        self.mask_e = np.tril_indices(data.shape[-1], -1)
        self.output_size = data.shape[-1]

    def f_loss(self, tr_inputs, targets, n_warmup):
        with self.model.param_precompute():
            if n_warmup > 0:
                self.model.update(np.arange(n_warmup))
            eeg_output = self.model.update(tr_inputs)
        loss_main = u.math.sqrt(u.math.mean((eeg_output - targets) ** 2))
        loss = 10. * loss_main + self.model.reg_loss()
        return loss, eeg_output

    @brainstate.transform.jit(
        static_argnums=(0, 3),
        static_argnames=['n_warmup']
    )
    def f_train(self, tr_inputs, targets, n_warmup: int):
        f_grad = brainstate.transform.grad(
            functools.partial(self.f_loss, n_warmup=n_warmup),
            self.weights, has_aux=True, return_value=True, check_states=False
        )
        grads, loss, eeg_output = f_grad(tr_inputs, targets)
        self.optimizer.step(grads)
        return loss, eeg_output

    @brainstate.transform.jit(static_argnums=0)
    def f_predict(self, inputs):
        with self.model.param_precompute():
            eeg_output, state_output = self.model(inputs, record_state=True)
        return eeg_output, state_output

    def train(self, inputs, n_epoches: int, n_warmup: int = 0):
        loss_his = []
        for i_epoch in range(n_epoches):
            self.model.init_all_states()
            loss, eeg_output = self.f_train(inputs, self.data, n_warmup=n_warmup)

            loss_np = np.asarray(loss)
            loss_his.append(loss_np)

            fc_emp = np.corrcoef(self.data, rowvar=False)
            fc_sim = np.corrcoef(eeg_output[10:, :], rowvar=False)
            cor = np.corrcoef(fc_sim[self.mask_e], fc_emp[self.mask_e])[0, 1]
            sim = np.diag(cosine_similarity(eeg_output.T, self.data.T)).mean()

            print(f'epoch = {i_epoch}, loss = {loss_np}, FC cor = {cor}, cos sim = {sim}')

        return np.array(loss_his)

    def test(self, inputs, n_warmup: int = 0):
        self.model.init_all_states()
        if n_warmup > 0:
            self.f_predict(np.zeros(n_warmup))
        eeg_output, state_output = self.f_predict(inputs)

        transient_num = 20
        fc = np.corrcoef(self.data, rowvar=False)
        fc_sim = np.corrcoef(eeg_output[transient_num:], rowvar=False)

        cor = np.corrcoef(fc_sim[self.mask_e], fc[self.mask_e])[0, 1]
        sim = np.diag(cosine_similarity(eeg_output.T, self.data.T)).mean()
        print(f'Testing FC = {cor}, cos_sim = {sim}')
        return eeg_output, state_output


def visualize_state_output(
    state_output,
    eeg_output,
    data_target,
    sc=None,
    node_indices=None,
    stimulus_window=None,
    mode='comprehensive',
    show_statistics=False,
    show=True
):
    """
    Visualize the state output from neural mass models with multiple visualization modes.

    Parameters
    ----------
    state_output : dict
        Dictionary containing state variables (e.g., 'M', 'E', 'I') with shape (time_steps, node_size)
    eeg_output : np.ndarray
        Predicted MEG/EEG signals with shape (time_steps, channels)
    data_target : np.ndarray
        Target MEG/EEG data with shape (time_steps, channels)
    sc : np.ndarray, optional
        Structural connectivity matrix (node_size, node_size). Required for 'representative' mode.
    node_indices : list, optional
        List of node indices to highlight. Default: automatically selected from sc if available,
        otherwise [2, 183, 5]
    stimulus_window : tuple, optional
        Tuple of (start_time, end_time) for stimulus window in milliseconds. Default: (100, 140)
    mode : str, optional
        Visualization mode: 'comprehensive' (12-panel overview), 'representative' (4-panel focused view),
        or 'both' (show both). Default: 'comprehensive'
    show_statistics : bool, optional
        Print statistical summary for representative nodes. Default: False
    show : bool, optional
        Display the figure(s). Default: True

    Returns
    -------
    fig or tuple of figs
        The created figure(s)
    """
    if stimulus_window is None:
        stimulus_window = (100, 140)

    stim_start, stim_end = stimulus_window

    # Extract states and convert to numpy
    state_output, eeg_output, data_target = jax.tree.map(
        np.asarray, (state_output, eeg_output, data_target)
    )
    keys = tuple(state_output.keys())

    time_steps = eeg_output.shape[0]
    node_size = state_output[keys[0]].shape[1]
    time_ms = np.arange(time_steps) * 1.0  # TR = 1ms

    # Set default node_indices if not provided
    if node_indices is None:
        if sc is not None:
            # Auto-select representative nodes based on connectivity
            node_degree = np.sum(sc, axis=1)
            # Select top 3 connected nodes
            node_indices = np.argsort(node_degree)[-3:].tolist()
        else:
            # Default fallback
            node_indices = [2, 183, 5] if node_size > 183 else list(range(min(3, node_size)))

    # Helper function: Select representative nodes
    def select_representative_nodes(sc_mat, primary_nodes, n_nodes=8):
        """Select representative nodes based on connectivity and spatial distribution."""
        node_degree = np.sum(sc_mat, axis=1)
        available = [i for i in range(node_size) if i not in primary_nodes]

        hub_idx = available[np.argmax(node_degree[available])]
        peripheral_idx = available[np.argmin(node_degree[available])]

        excluded = set(primary_nodes + [hub_idx, peripheral_idx])
        spatial_candidates = [i for i in range(node_size) if i not in excluded]

        # Select spatially distributed nodes
        early_spatial = spatial_candidates[np.argmin(np.abs(np.array(spatial_candidates) - node_size * 0.15))]
        middle_spatial = spatial_candidates[np.argmin(np.abs(np.array(spatial_candidates) - node_size * 0.47))]
        late_spatial = spatial_candidates[np.argmin(np.abs(np.array(spatial_candidates) - node_size * 0.80))]

        indices = primary_nodes + [hub_idx, peripheral_idx, early_spatial, middle_spatial, late_spatial]
        labels = [
            f'Primary-1 (N{primary_nodes[0]})',
            f'Primary-2 (N{primary_nodes[1]})',
            f'Primary-3 (N{primary_nodes[2]})',
            f'Hub (N{hub_idx})',
            f'Peripheral (N{peripheral_idx})',
            f'Early (N{early_spatial})',
            f'Middle (N{middle_spatial})',
            f'Late (N{late_spatial})'
        ]
        types = ['primary', 'primary', 'primary', 'hub', 'peripheral', 'spatial', 'spatial', 'spatial']
        colors = ['darkred', 'red', 'lightcoral', 'blue', 'green', 'darkgray', 'gray', 'lightgray']

        return {'indices': indices, 'labels': labels, 'types': types, 'colors': colors}

    # Helper function: Print statistics
    def print_statistics(node_info):
        """Print statistical summary of state values."""
        indices = node_info['indices']
        labels = node_info['labels']

        print("\n" + "=" * 80)
        print("REPRESENTATIVE NODE STATE STATISTICS")
        print("=" * 80)

        for node_idx, label in zip(indices, labels):
            print(f"\n{label}:")
            print("-" * 60)

            for state_name in keys:
                state = state_output[state_name][:, node_idx]
                peak_val = u.math.max(u.math.abs(state))
                peak_time = u.math.argmax(u.math.abs(state))
                mean_val = u.math.mean(state)
                std_val = u.math.std(state)

                print(f"  {state_name} State:")
                print(f"    Peak Value: {peak_val:>10.4f}  (at t={peak_time} ms)")
                print(f"    Mean:       {mean_val:>10.4f}  ±{std_val:.4f}")

        print("\n" + "=" * 80 + "\n")

    # COMPREHENSIVE VISUALIZATION
    def create_comprehensive_view():
        """Create 12-panel comprehensive visualization."""
        fig = plt.figure(figsize=(18, 16))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Row 1: Time series plots for selected nodes
        colors = plt.cm.viridis(np.linspace(0, 1, len(node_indices)))
        for col, (name, state) in enumerate(state_output.items()):
            state = u.get_magnitude(state)
            ax = fig.add_subplot(gs[0, col])
            for idx, node_idx in enumerate(node_indices):
                ax.plot(time_ms, state[:, node_idx], label=f'Node {node_idx}', color=colors[idx], linewidth=1.5)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('State Value')
            ax.set_title(f'{name} - Selected Nodes')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.axvspan(stim_start, stim_end, alpha=0.2, color='red', label='Stimulus')

        # Row 2: Heatmaps for all nodes
        for col, (name, state) in enumerate(state_output.items()):
            state = u.get_magnitude(state)
            ax = fig.add_subplot(gs[1, col])
            im = ax.imshow(state.T, aspect='auto', cmap='RdBu_r', extent=(0, time_steps, 0, node_size), origin='lower')
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Node Index')
            ax.set_title(f'{name} - All Nodes')
            ax.axvline(stim_start, color='yellow', linestyle='--', linewidth=1, alpha=0.7)
            ax.axvline(stim_end, color='yellow', linestyle='--', linewidth=1, alpha=0.7)
            plt.colorbar(im, ax=ax, label='State Value')

        # Row 3: Distributions of final state values
        for col, (name, state) in enumerate(state_output.items()):
            state = u.get_magnitude(state)
            ax = fig.add_subplot(gs[2, col])
            st = state[-1, :]
            ax.hist(st, bins=40, color='steelblue', alpha=0.7, edgecolor='black')
            ax.axvline(st.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {st.mean():.3f}')
            ax.axvline(st.mean() + st.std(), color='orange', linestyle=':', linewidth=2, alpha=0.7)
            ax.axvline(st.mean() - st.std(), color='orange', linestyle=':',
                       linewidth=2, alpha=0.7, label=f'Std: {st.std():.3f}')
            ax.set_xlabel('Final State Value')
            ax.set_ylabel('Count')
            ax.set_title(f'{name} - Final State Distribution')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')

        # Row 4: MEG comparison
        n_channels = min(3, eeg_output.shape[1])
        channel_indices = np.linspace(0, eeg_output.shape[1] - 1, n_channels, dtype=int)

        for col, ch_idx in enumerate(channel_indices):
            ax = fig.add_subplot(gs[3, col])
            ax.plot(time_ms, data_target[:, ch_idx], 'k--', linewidth=2, label='Target', alpha=0.7)
            ax.plot(time_ms, eeg_output[:, ch_idx], 'b-', linewidth=1.5, label='Predicted')

            corr = np.corrcoef(data_target[:, ch_idx], eeg_output[:, ch_idx])[0, 1]
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('MEG Signal')
            ax.set_title(f'Channel {ch_idx} (corr: {corr:.3f})')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.axvspan(stim_start, stim_end, alpha=0.2, color='red')

        plt.suptitle('Neural Mass Model State Dynamics - Comprehensive View',
                     fontsize=16, fontweight='bold', y=0.995)
        return fig

    # REPRESENTATIVE VISUALIZATION
    def create_representative_view(node_info):
        """Create 4-panel representative nodes visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        indices = node_info['indices']
        labels = node_info['labels']
        colors = node_info['colors']

        # Panel 1-3: State trajectories for P, E, I
        for panel_idx in range(3):
            ax = axes.flat[panel_idx]
            state_name = keys[panel_idx] if panel_idx < len(keys) else keys[0]
            state = state_output[state_name]
            state = u.get_magnitude(state)

            for idx, node_idx in enumerate(indices):
                ax.plot(time_ms, state[:, node_idx], label=labels[idx], color=colors[idx], linewidth=2.0)
            ax.axvspan(stim_start, stim_end, alpha=0.2, color='red', zorder=0)
            ax.set_xlabel('Time (ms)', fontsize=11)
            ax.set_ylabel('State Value', fontsize=11)
            ax.set_title(f'{state_name} State - Representative Nodes', fontsize=13, fontweight='bold')
            ax.legend(fontsize=8, loc='best', ncol=2)
            ax.grid(True, alpha=0.3, color='lightgray')

        # Panel 4: Combined P-E-I for primary auditory node
        ax = axes[1, 1]
        primary_node = indices[0]
        line_styles = ['-', '--', '-.']
        state_colors = ['blue', 'red', 'green']

        for idx, (state_name, style, color) in enumerate(zip(keys, line_styles, state_colors)):
            state = state_output[state_name]
            state = u.get_magnitude(state)
            ax.plot(time_ms, state[:, primary_node], label=f'{state_name} State',
                    linestyle=style, color=color, linewidth=2.5)

        ax.axvspan(stim_start, stim_end, alpha=0.2, color='red', zorder=0)
        ax.set_xlabel('Time (ms)', fontsize=11)
        ax.set_ylabel('State Value', fontsize=11)
        ax.set_title(f'Combined Dynamics in Primary Node {primary_node}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3, color='lightgray')

        plt.suptitle('Representative Brain Region State Dynamics', fontsize=16, fontweight='bold', y=0.995)
        return fig

    # Main execution logic
    if mode == 'comprehensive':
        fig = create_comprehensive_view()
        if show:
            plt.show()
        return fig

    elif mode == 'representative':
        if sc is None:
            raise ValueError("Structural connectivity matrix 'sc' is required for representative mode")
        node_info = select_representative_nodes(sc, node_indices)
        if show_statistics:
            print_statistics(node_info)
        fig = create_representative_view(node_info)
        if show:
            plt.show()
        return fig

    elif mode == 'both':
        if sc is None:
            raise ValueError("Structural connectivity matrix 'sc' is required for representative mode")
        node_info = select_representative_nodes(sc, node_indices)
        if show_statistics:
            print_statistics(node_info)

        fig_comp = create_comprehensive_view()
        fig_rep = create_representative_view(node_info)

        if show:
            plt.show()
        return fig_comp, fig_rep

    else:
        raise ValueError(f"Invalid mode: {mode}. Choose 'comprehensive', 'representative', or 'both'")


def train_language_horn():
    brainstate.environ.set(dt=1.0 * u.ms)

    lm, sc, dist, data_verb, uu = get_language_data()
    uu *= 5.0

    model = HORNNetworkTR(
        dist.shape[0],
        sc=sc, dist=dist, mu=0.1, lm=lm, cy0=Const(5),
        y0=Param(-0.5, reg=GaussianReg(-0.5, 0.05, fit_hyper=True)),
    )
    fitting = ModelFitting(model, data_verb, braintools.optim.Adam(lr=5e-3))
    fitting.train(uu, n_epoches=1500)
    eeg_output, state_output = fitting.test(uu)

    # Visualize with representative mode
    visualize_state_output(state_output, eeg_output, data_verb, sc=sc, mode='both', show=True)


def train_language_jr():
    lm, sc, dist, data_verb, uu = get_language_data()
    uu *= 5.0

    model = JansenRitNetworkTR(
        dist.shape[0],
        sc=sc, dist=dist, mu=1., lm=lm, cy0=Const(5),
        y0=Param(-0.5, reg=GaussianReg(-0.5, 0.05, fit_hyper=True)),
    )
    fitting = ModelFitting(model, data_verb, braintools.optim.Adam(lr=5e-2))
    fitting.train(uu, n_epoches=100)
    eeg_output, state_output = fitting.test(uu)

    # Visualize with representative mode
    visualize_state_output(state_output, eeg_output, data_verb, sc=sc, mode='both', show=True)


def train_hdeeg_jr():
    # lm, sc, dist, data_verb, uu = get_hdeeg_data()
    lm, sc, dist, data_verb, uu = get_hdeeg_data_v2()
    uu *= 5.0

    # Extract stimulus information from uu
    # Find which nodes received stimulus (non-zero values)
    stim_nodes = np.where(np.any(uu != 0, axis=0))[0]
    # Find time window of stimulus
    stim_times = np.where(np.any(uu != 0, axis=1))[0]
    stim_window = (stim_times[0], stim_times[-1]) if len(stim_times) > 0 else (65, 75)
    # Select top 3 stimulated nodes
    node_indices = stim_nodes[:3].tolist() if len(stim_nodes) >= 3 else stim_nodes.tolist()

    model = JansenRitNetworkTR(
        dist.shape[0],
        sc=sc, dist=dist, mu=1., lm=lm, cy0=Const(5),
        y0=Param(-0.5, reg=GaussianReg(-0.5, 0.05, fit_hyper=True)),
    )
    fitting = ModelFitting(model, data_verb, braintools.optim.Adam(lr=5e-2))
    fitting.train(uu, n_epoches=100)
    eeg_output, state_output = fitting.test(uu)

    # Visualize with representative mode
    visualize_state_output(
        state_output, eeg_output, data_verb,
        sc=sc, node_indices=node_indices, stimulus_window=stim_window, mode='both', show=True
    )


if __name__ == '__main__':
    pass
    # train_language_horn()
    # train_language_jr()
    train_hdeeg_jr()
