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
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils import utils
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.utils.wyx_utils import *
import cv2
from igibson.utils.mesh_util import *


def main(selection="user", headless=False, short_exec=False):
    # s = Simulator(mode="headless", use_pb_gui=True)
    s = Simulator(gravity=0,
                  image_width=500, image_height=500,
                  use_pb_gui=False)
    scene = EmptyScene(floor_plane_rgba=[0.6, 0.6, 0.6, 1])
    # scene = InteractiveIndoorScene(
    #     "Beechwood_0_int",
    #     # load_object_categories=[],  # To load only the building. Fast
    #     build_graph=True,
    # )
    s.import_scene(scene)

    shapenet_filename = '/remote-home/2132917/Desktop/ShapeNet_dataset/ShapeNetCore.v2/03001627/1a6f615e8b1b5ae4dbbc9440457e303e/models/model_normalized.obj'
    shapenet = ShapeNetObject(path=shapenet_filename, position=[1, 2, 3])

    s.import_object(shapenet)

    # Main simulation loop
    try:
        # Main recording loop
        while True:
            # Step simulation.
            s.step()
            frame = s.renderer.render(modes=('rgb'))
            render_images = cv2.cvtColor(np.concatenate(frame, axis=1), cv2.COLOR_RGB2BGR)
            # render_images = cv2.resize(render_images, dsize=(600, 100))

            # shapenet_poses = shapenet.get_poses()
            # shapenet_bb_center, shapenet_bb, shapenet_bb_coordinate = shapenet.update_center_point_bounding_box()
            #
            # # 绘制物体坐标系xyz
            # camera_intrinsics = s.renderer.get_intrinsics()
            # camera_extrinsics = compute_camera_extrinsics_matrix(s)
            # pose_xy = compute_point_pixel_plane_coordinates(shapenet_bb_center, camera_extrinsics, camera_intrinsics)
            # # pose_xy[0] = 500 - pose_xy[0]
            # shapenet_bb_center_xy = pose_xy
            # coordinate_xyz = []
            # for coordinate in shapenet_bb_coordinate:
            #     camera_intrinsics = s.renderer.get_intrinsics()
            #     camera_extrinsics = compute_camera_extrinsics_matrix(s)
            #     pose_xy = compute_point_pixel_plane_coordinates(coordinate, camera_extrinsics, camera_intrinsics)
            #     # pose_xy[0] = 500 - pose_xy[0]
            #     coordinate_xyz.append(pose_xy)
            # # cv2.line(render_images, [int(shapenet_bb_center_xy[0]), int(shapenet_bb_center_xy[1])],
            # #          [int(coordinate_xyz[0][0]), int(coordinate_xyz[0][1])], (0, 0, 255), 3)
            # # cv2.line(render_images, [int(shapenet_bb_center_xy[0]), int(shapenet_bb_center_xy[1])],
            # #          [int(coordinate_xyz[1][0]), int(coordinate_xyz[1][1])], (0, 255, 0), 3)
            # # cv2.line(render_images, [int(shapenet_bb_center_xy[0]), int(shapenet_bb_center_xy[1])],
            # #          [int(coordinate_xyz[2][0]), int(coordinate_xyz[2][1])], (255, 0, 0), 3)
            #
            # shapenet_bb_xy = []
            # bbxy_x_max, bbxy_x_min, bbxy_y_max, bbxy_y_min = -1000, 1000, -1000, 1000
            # for points in shapenet_bb:
            #     camera_intrinsics = s.renderer.get_intrinsics()
            #     camera_extrinsics = compute_camera_extrinsics_matrix(s)
            #     pose_xy = compute_point_pixel_plane_coordinates(points, camera_extrinsics, camera_intrinsics)
            #     # pose_xy[0] = 500 - pose_xy[0]
            #     shapenet_bb_xy.append(pose_xy)
            #
            #     if pose_xy[0] > bbxy_x_max:
            #         bbxy_x_max = pose_xy[0]
            #     if pose_xy[0] < bbxy_x_min:
            #         bbxy_x_min = pose_xy[0]
            #     if pose_xy[1] > bbxy_y_max:
            #         bbxy_y_max = pose_xy[1]
            #     if pose_xy[1] < bbxy_y_min:
            #         bbxy_y_min = pose_xy[1]
            #
            # # 绘制立体框的8个顶点
            # for point in shapenet_bb_xy:
            #     cv2.circle(render_images, [int(point[0][0]), int(point[1][0])], 4, (0, 140, 255), -1)
            # # 绘制2d边界框
            # cv2.rectangle(render_images, (int(bbxy_x_min), int(bbxy_y_min)), (int(bbxy_x_max), int(bbxy_y_max)),
            #               (255, 0, 255), 1)

            body_ids = s.scene.get_body_ids()
            print(s.renderer.visual_objects)
            print(s.renderer.instance_id_to_pb_id)
            # objects = s.scene.objects_by_name
            print(body_ids)
            # s.scene.get_objects()
            # body_ids = [7]
            for id in body_ids:
                try:
                    trans, orn = p.getBasePositionAndOrientation(id)
                    orn = quat2rotmat(xyzw2wxyz(orn))

                    o_3d_oral = (0, 0, 0)
                    x_3d_oral = (0.1, 0, 0)
                    y_3d_oral = (0, 0.1, 0)
                    z_3d_oral = (0, 0, 0.1)

                    o_3d = np.matmul(orn, np.array(
                        [[o_3d_oral[0]], [o_3d_oral[1]], [o_3d_oral[2]], [1]], dtype=float)) + \
                           np.matrix([[trans[0]], [trans[1]], [trans[2]], [0]])
                    x_3d = np.matmul(orn, np.array(
                        [[x_3d_oral[0]], [x_3d_oral[1]], [x_3d_oral[2]], [1]], dtype=float)) + \
                           np.matrix([[trans[0]], [trans[1]], [trans[2]], [0]])
                    y_3d = np.matmul(orn, np.array(
                        [[y_3d_oral[0]], [y_3d_oral[1]], [y_3d_oral[2]], [1]], dtype=float)) + \
                           np.matrix([[trans[0]], [trans[1]], [trans[2]], [0]])
                    z_3d = np.matmul(orn, np.array(
                        [[z_3d_oral[0]], [z_3d_oral[1]], [z_3d_oral[2]], [1]], dtype=float)) + \
                           np.matrix([[trans[0]], [trans[1]], [trans[2]], [0]])

                    # 绘制物体坐标系xyz
                    camera_intrinsics = s.renderer.get_intrinsics()
                    camera_extrinsics = compute_camera_extrinsics_matrix(s)
                    o_2d = trans_3d_point_to_2d_pixel(o_3d, camera_extrinsics, camera_intrinsics)
                    x_2d = trans_3d_point_to_2d_pixel(x_3d, camera_extrinsics, camera_intrinsics)
                    y_2d = trans_3d_point_to_2d_pixel(y_3d, camera_extrinsics, camera_intrinsics)
                    z_2d = trans_3d_point_to_2d_pixel(z_3d, camera_extrinsics, camera_intrinsics)

                    whether_in_the_image = True
                    for i in [int(o_2d[0]), int(o_2d[1]), int(x_2d[0]), int(x_2d[1]), int(y_2d[0]), int(y_2d[1]),
                              int(z_2d[0]), int(z_2d[1])]:
                        whether_in_the_image = whether_in_the_image and i > 0 and i < 500
                    if whether_in_the_image == True:
                        cv2.line(render_images, [int(o_2d[0]), int(o_2d[1])],
                                 [int(x_2d[0]), int(x_2d[1])], (0, 0, 255), 3)
                        cv2.line(render_images, [int(o_2d[0]), int(o_2d[1])],
                                 [int(y_2d[0]), int(y_2d[1])], (0, 255, 0), 3)
                        cv2.line(render_images, [int(o_2d[0]), int(o_2d[1])],
                                 [int(z_2d[0]), int(z_2d[1])], (255, 0, 0), 3)
                except Exception as e:
                    print(e)
            # cv2.imshow("object coordinate", render_images)

            # # 绘制世界坐标系
            # o_3d_oral = (0, 0, 0, 1)
            # x_3d_oral = (10, 0, 0, 1)
            # y_3d_oral = (0, 10, 0, 1)
            # z_3d_oral = (0, 0, 10, 1)
            # camera_intrinsics = s.renderer.get_intrinsics()
            # camera_extrinsics = compute_camera_extrinsics_matrix(s)
            # o_2d = compute_point_pixel_plane_coordinates(o_3d_oral, camera_extrinsics, camera_intrinsics)
            # x_2d = compute_point_pixel_plane_coordinates(x_3d_oral, camera_extrinsics, camera_intrinsics)
            # y_2d = compute_point_pixel_plane_coordinates(y_3d_oral, camera_extrinsics, camera_intrinsics)
            # z_2d = compute_point_pixel_plane_coordinates(z_3d_oral, camera_extrinsics, camera_intrinsics)
            # cv2.line(render_images, [int(o_2d[0]), int(o_2d[1])],
            #          [int(x_2d[0]), int(x_2d[1])], (0, 0, 255), 3)
            # cv2.line(render_images, [int(o_2d[0]), int(o_2d[1])],
            #          [int(y_2d[0]), int(y_2d[1])], (0, 255, 0), 3)
            # cv2.line(render_images, [int(o_2d[0]), int(o_2d[1])],
            #          [int(z_2d[0]), int(z_2d[1])], (255, 0, 0), 3)

            cv2.imshow("render_images", render_images)
            # cv2.waitKey(1)


    finally:
        s.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
