import logging
from sys import platform

import numpy as np
import cv2

from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.render.profiler import Profiler
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.scenes.empty_scene import EmptyScene
from igibson.scenes.scene_base import Scene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_available_ig_scenes
from igibson.utils.utils import let_user_pick
import pybullet as p
from igibson.utils.wyx_utils import *
from igibson.utils.mesh_util import *
import time
import math
from scipy.spatial.transform import Rotation


def main():
    scene_id = 'scene_test01'
    settings = MeshRendererSettings(enable_shadow=True, msaa=True, blend_highlight=True, )
    # if platform == "darwin":
    #     settings.texture_scale = 0.5
    s = Simulator(
        mode="gui_interactive",
        image_width=500,
        image_height=500,
        rendering_settings=settings,
        # use_pb_gui=True
    )

    scene = InteractiveIndoorScene(
        scene_id,
        urdf_file="scene_test03",
        # load_object_categories=[],  # To load only the building. Fast
        build_graph=True,
    )
    # scene = EmptyScene()
    # print(scene.get_objects())
    s.import_scene(scene)

    step = 0
    step_size = 200
    a = 2.2
    b = 1.5
    x_list = np.linspace(-1.5, 1.5, step_size)
    print(x_list)
    target_x_list = np.linspace(0.2, -0.2, step_size)
    print(target_x_list)
    pose_truth = list()
    while step != step_size:
        # while True:
        with Profiler("Simulator step"):
            # 设置相机位姿和相机观察目标点
            x = x_list[step]
            y = (b) ** 2 * math.sqrt(1 - (x / a) ** 2)
            z = 0.5 + 0.25 * math.sin(x * 3.14 * 2.5)
            camera_pose = [x, y, z]

            target_x = target_x_list[step]
            target_y = (0.2) ** 2 - (target_x) ** 2
            target_z = 0 + 0.1 * math.sin(x * 3.14 * 4)
            camera_target = [target_x, target_y, target_z]

            s.viewer.px = camera_pose[0]
            s.viewer.py = camera_pose[1]
            s.viewer.pz = camera_pose[2]
            s.viewer.view_direction = np.array(
                [camera_target[0] - camera_pose[0], camera_target[1] - camera_pose[1],
                 camera_target[2] - camera_pose[2]])
            s.viewer.view_direction = s.viewer.view_direction / np.sqrt(np.sum(s.viewer.view_direction ** 2))

            s.step()
            step += 1

            # 输出相机初始位姿
            if step == 1:
                print(compute_camera_extrinsics_matrix(s))
            T = compute_camera_extrinsics_matrix(s)
            T = np.linalg.inv(T)
            R = T[:3, :3]
            R_quat = Rotation.from_matrix(R).as_quat()
            pose_truth.append([step, T[0][3], T[1][3], T[2][3], R_quat[0], R_quat[1], R_quat[2], R_quat[3]])

            frame = s.renderer.render(modes='rgb')
            render_images = cv2.cvtColor(np.concatenate(frame, axis=1), cv2.COLOR_RGB2BGR)
            rgb_image = cv2.cvtColor(np.concatenate(frame, axis=1), cv2.COLOR_RGB2BGR)
            # 渲染深度图
            frame_3d = s.renderer.render(modes="3d")[0]
            depth_low = 0
            depth_high = 20
            depth = -frame_3d[:, :, 2:3]
            # 0.0 is a special value for invalid entries
            depth[depth < depth_low] = 0.0
            depth[depth > depth_high] = 0.0
            # re-scale depth to [0.0, 1.0]
            depth /= depth_high

            body_ids = s.scene.get_body_ids()
            objects = s.scene.objects_by_name
            # print(s.renderer.get_visual_objects())
            # print(body_ids)
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

                    camera_intrinsics = s.renderer.get_intrinsics()
                    camera_extrinsics = compute_camera_extrinsics_matrix(s)
                    o_2d = trans_3d_point_to_2d_pixel(o_3d, camera_extrinsics, camera_intrinsics)
                    x_2d = trans_3d_point_to_2d_pixel(x_3d, camera_extrinsics, camera_intrinsics)
                    y_2d = trans_3d_point_to_2d_pixel(y_3d, camera_extrinsics, camera_intrinsics)
                    z_2d = trans_3d_point_to_2d_pixel(z_3d, camera_extrinsics, camera_intrinsics)

                    whether_in_the_image = True
                    for i in [int(o_2d[0]), int(o_2d[1]), int(x_2d[0]), int(x_2d[1]), int(y_2d[0]), int(y_2d[1]),
                              int(z_2d[0]), int(z_2d[1])]:
                        whether_in_the_image = whether_in_the_image and 0 < i < 500
                    if whether_in_the_image == True:
                        cv2.line(render_images, [int(o_2d[0]), int(o_2d[1])],
                                 [int(x_2d[0]), int(x_2d[1])], (0, 0, 255), 3)
                        cv2.line(render_images, [int(o_2d[0]), int(o_2d[1])],
                                 [int(y_2d[0]), int(y_2d[1])], (0, 255, 0), 3)
                        cv2.line(render_images, [int(o_2d[0]), int(o_2d[1])],
                                 [int(z_2d[0]), int(z_2d[1])], (255, 0, 0), 3)
                except Exception as e:
                    print(e)

            # 绘制世界坐标系
            try:
                camera_intrinsics = s.renderer.get_intrinsics()
                camera_extrinsics = compute_camera_extrinsics_matrix(s)
                o_2d = trans_3d_point_to_2d_pixel([0, 0, 0, 1], camera_extrinsics, camera_intrinsics)
                x_2d = trans_3d_point_to_2d_pixel([1, 0, 0, 1], camera_extrinsics, camera_intrinsics)
                y_2d = trans_3d_point_to_2d_pixel([0, 1, 0, 1], camera_extrinsics, camera_intrinsics)
                z_2d = trans_3d_point_to_2d_pixel([0, 0, 1, 1], camera_extrinsics, camera_intrinsics)
                whether_in_the_image = True
                for i in [int(o_2d[0]), int(o_2d[1]), int(x_2d[0]), int(x_2d[1]), int(y_2d[0]), int(y_2d[1]),
                          int(z_2d[0]), int(z_2d[1])]:
                    whether_in_the_image = whether_in_the_image and 0 < i < 500
                if whether_in_the_image == True:
                    cv2.line(render_images, [int(o_2d[0]), int(o_2d[1])],
                             [int(x_2d[0]), int(x_2d[1])], (0, 0, 255), 8)
                    cv2.line(render_images, [int(o_2d[0]), int(o_2d[1])],
                             [int(y_2d[0]), int(y_2d[1])], (0, 255, 0), 8)
                    cv2.line(render_images, [int(o_2d[0]), int(o_2d[1])],
                             [int(z_2d[0]), int(z_2d[1])], (255, 0, 0), 8)
            except Exception as e:
                print(e)

            # 转化为8bit离散数据，避免保存的图片为黑色
            rgb_image = rgb_image * 255
            rgb_image = rgb_image.astype(np.uint8)
            depth = depth * 255
            depth = depth.astype(np.uint8)

            cv2.imshow("object coordinate", render_images)
            cv2.imshow("depth image", depth)

            cv2.imwrite("/home/ubuntu/Desktop/iGibson_study/igibson_dataset/easy01/rgb/" + str(step) + ".png",
                        rgb_image)
            cv2.imwrite("/home/ubuntu/Desktop/iGibson_study/igibson_dataset/easy01/depth/" + str(step) + ".png", depth)
            time.sleep(0.05)

    s.disconnect()

    with open('/home/ubuntu/Desktop/iGibson_study/igibson_dataset/easy01/pose_truth.txt', 'w') as f:
        for line in pose_truth:
            line = ' '.join([str(i) for i in line])
            f.write(line)
            f.write('\n')


if __name__ == "__main__":
    main()
