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


def compute_point_pixel_plane_coordinates(pose_w, extrinsics, intrinsics):
    pose_c = np.matmul(extrinsics, np.array([[pose_w[0]], [pose_w[1]], [pose_w[2]], [1]], dtype=float))
    pose_c = pose_c / pose_c[3]
    pose_c = pose_c[:3]
    pose_xy = np.matmul(intrinsics, pose_c)
    pose_xy = pose_xy / pose_xy[2]
    pose_xy = pose_xy[:2]
    return pose_xy


def compute_camera_extrinsics_matrix(s):
    camera_position = [s.viewer.px, s.viewer.py, s.viewer.pz]
    up = s.renderer.up

    direction = camera_position - s.renderer.target
    direction = direction / np.sqrt(np.sum(direction ** 2))
    xaxis = np.cross(up, direction)
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


def main(selection="user", headless=False, short_exec=False):
    # s = Simulator(mode="headless", use_pb_gui=True)
    s = Simulator(gravity=0,
                  image_width=500, image_height=500,
                  use_pb_gui=False)
    scene = EmptyScene(floor_plane_rgba=[0.6, 0.6, 0.6, 1])
    s.import_scene(scene)

    # shapenet_filename = '/remote-home/2132917/Desktop/ShapeNet_dataset/ShapeNetCore.v2/03001627/1a6f615e8b1b5ae4dbbc9440457e303e/models/model_normalized.obj'
    # shapenet = ShapeNetObject(path=shapenet_filename, position=[1, 2, 3])
    # cube = Cube(pos=[1, 2, 3], dim=[1, 2, 3], visual_only=False, mass=1, color=[0.5, 1, 1, 0.2])
    marker = VisualMarker(visual_shape=p.GEOM_BOX,
                          rgba_color=[0.5, 0, 0, 0.1],
                          # radius=1.0,
                          half_extents=[1, 1, 1],
                          length=0,
                          initial_offset=[0, 0, 0],
                          filename=None,
                          scale=[1.0], )

    # s.import_object(shapenet)
    # s.import_object(cube)
    s.import_object(marker)

    marker.set_position_orientation([1, 2, 3], Rotation.from_euler("XYZ", [0, 0, 0]).as_quat())

    # Main simulation loop
    try:
        steps = 0
        max_steps = -1 if not short_exec else 1000

        # Main recording loop
        while steps != max_steps:
            # Step simulation.
            s.step()
            # shapenet.update_center_point_bounding_box()

            camera_intrinsics = s.renderer.get_intrinsics()
            camera_extrinsics = compute_camera_extrinsics_matrix(s)
            pose_xy = compute_point_pixel_plane_coordinates([1, 2, 3], camera_extrinsics, camera_intrinsics)
            print(pose_xy)

            # print(s.viewer.px, s.viewer.py, s.viewer.pz)
            # print(s.viewer.view_direction)
            # print(s.viewer.theta, s.viewer.phi)
            # print(s.renderer.get_intrinsics())
            # print(s.renderer.camera)
            # print(s.renderer.target)
            # print(s.renderer.up)
            # print(s.renderer.V)

            steps += 1
    finally:
        s.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
