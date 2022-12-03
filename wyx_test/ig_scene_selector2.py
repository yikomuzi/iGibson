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


def main(selection="user", headless=False, short_exec=False):
    """
    Prompts the user to select any available interactive scene and loads it.
    Shows how to load directly scenes without the Environment interface
    Shows how to sample points in the scene by room type and how to compute geodesic distance and the shortest path
    """
    # print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    # available_ig_scenes = get_first_options()
    # scene_id = available_ig_scenes[let_user_pick(available_ig_scenes, selection=selection) - 1]
    scene_id = 'Rs_int'
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
        # load_object_categories=[],  # To load only the building. Fast
        build_graph=True,
    )
    # scene = EmptyScene()
    print(scene.get_objects())
    s.import_scene(scene)

    print(s.renderer)

    step = 0
    while True:
        with Profiler("Simulator step"):
            # s.viewer.view_direction = [s.viewer.view_direction[0], s.viewer.view_direction[1] - 0.1,
            # s.viewer.view_direction[2]]
            s.step()
            step += 1

            frame = s.renderer.render(modes=('rgb'))
            render_images = cv2.cvtColor(np.concatenate(frame, axis=1), cv2.COLOR_RGB2BGR)

            body_ids = s.scene.get_body_ids()
            objects = s.scene.objects_by_name
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
            cv2.imshow("object coordinate", render_images)

    s.disconnect()


# def get_first_options():
#     return get_available_ig_scenes()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
