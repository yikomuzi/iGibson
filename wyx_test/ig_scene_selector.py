import logging
from sys import platform

import numpy as np
import cv2

from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.render.profiler import Profiler
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_available_ig_scenes
from igibson.utils.utils import let_user_pick


def main(selection="user", headless=False, short_exec=False):
    """
    Prompts the user to select any available interactive scene and loads it.
    Shows how to load directly scenes without the Environment interface
    Shows how to sample points in the scene by room type and how to compute geodesic distance and the shortest path
    """
    # print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    # available_ig_scenes = get_first_options()
    # scene_id = available_ig_scenes[let_user_pick(available_ig_scenes, selection=selection) - 1]
    scene_id = 'Beechwood_0_int'
    settings = MeshRendererSettings(enable_shadow=True, msaa=True, blend_highlight=True, )
    # if platform == "darwin":
    #     settings.texture_scale = 0.5
    s = Simulator(
        mode="gui_interactive",
        image_width=100,
        image_height=100,
        rendering_settings=settings,
        # use_pb_gui=True
    )

    scene = InteractiveIndoorScene(
        scene_id,
        # load_object_categories=[],  # To load only the building. Fast
        build_graph=True,
    )
    print(scene.get_objects())
    s.import_scene(scene)

    print(s.renderer)

    # Shows how to sample points in the scene
    # np.random.seed(0)
    # for _ in range(10):
    #     pt = scene.get_random_point_by_room_type("living_room")[1]
    #     print("Random point in living_room: {}".format(pt))
    #
    # for _ in range(10):
    #     random_floor = scene.get_random_floor()
    #     p1 = scene.get_random_point(random_floor)[1]
    #     p2 = scene.get_random_point(random_floor)[1]
    #     shortest_path, geodesic_distance = scene.get_shortest_path(random_floor, p1[:2], p2[:2], entire_path=True)
    #     print("Random point 1: {}".format(p1))
    #     print("Random point 2: {}".format(p2))
    #     print("Geodesic distance between p1 and p2: {}".format(geodesic_distance))
    #     print("Shortest path from p1 to p2: {}".format(shortest_path))
    #
    # if not headless:
    #     input("Press enter")

    # max_steps = -1 if not short_exec else 1000

    # s.renderer.camera = [0, 0, 0]

    # s.viewer.px = 0
    # s.viewer.py = 0
    # s.viewer.pz = 1
    # s.viewer.view_direction = [0, 0, np.pi / 4]

    step = 0
    while True:
        with Profiler("Simulator step"):
            # s.viewer.view_direction = [s.viewer.view_direction[0], s.viewer.view_direction[1] - 0.1,
            # s.viewer.view_direction[2]]
            s.step()
            step += 1

            # a = s.renderer.transform_pose(np.concatenate([[0, 1, 1], [0.924, 0.383, 0, 0]]))

            print(s.renderer.get_intrinsics())
            print(s.viewer.px, s.viewer.py, s.viewer.pz)
            print(s.viewer.view_direction)
            # print(s.viewer.theta, s.viewer.phi)
            # print(s.renderer.get_intrinsics())
            print(s.renderer.camera)
            # print(s.renderer.target)
            # print(s.renderer.up)
            print(s.renderer.V)
            print(s.renderer.P)
            frame = s.renderer.render(modes=('rgb', 'normal', 'seg', '3d', 'scene_flow', 'optical_flow'))
            render_images = cv2.cvtColor(np.concatenate(frame, axis=1), cv2.COLOR_RGB2BGR)
            render_images = cv2.resize(render_images, dsize=(600, 100))
            cv2.imshow("render_images", render_images)

    s.disconnect()


# def get_first_options():
#     return get_available_ig_scenes()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
