import itertools
import logging
import os

import numpy as np
import pybullet as p
import trimesh
from scipy.spatial.transform import Rotation

import igibson
from igibson.objects.articulated_object import URDFObject
from igibson.objects.shapenet_object import ShapeNetObject
from igibson.objects.visual_marker import VisualMarker
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils import utils


# 已知一个三维点，求其在像素平面的坐标
def trans_3d_point_to_2d_pixel(pose_w, extrinsics, intrinsics):
    pose_c = np.matmul(extrinsics, np.array([[pose_w[0]], [pose_w[1]], [pose_w[2]], [1]], dtype=float))
    pose_c = pose_c / pose_c[3]
    pose_c = pose_c[:3]
    print(pose_c)
    if pose_c[2] < 0:
        e = Exception("深度值为负数")
        raise e
    pose_xy = np.matmul(intrinsics, pose_c)
    pose_xy = pose_xy / pose_xy[2]
    pose_xy = pose_xy[:2]
    return pose_xy


def compute_camera_extrinsics_matrix(s):
    camera_position = [s.viewer.px, s.viewer.py, s.viewer.pz]
    up = s.renderer.up

    direction = -(camera_position - s.renderer.target)
    direction = direction / np.sqrt(np.sum(direction ** 2))
    xaxis = -np.cross(up, direction)
    xaxis = xaxis / np.sqrt(np.sum(xaxis ** 2))
    yaxis = np.cross(direction, xaxis)
    yaxis = yaxis / np.sqrt(np.sum(yaxis ** 2))

    r = np.array([[xaxis[0], yaxis[0], direction[0]],
                  [xaxis[1], yaxis[1], direction[1]],
                  [xaxis[2], yaxis[2], direction[2]]])

    t = np.array([[r[0][0], r[0][1], r[0][2], camera_position[0]],
                  [r[1][0], r[1][1], r[1][2], camera_position[1]],
                  [r[2][0], r[2][1], r[2][2], camera_position[2]],
                  [0, 0, 0, 1]])
    t = np.linalg.inv(t)
    return t
