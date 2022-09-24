import itertools
import logging
import os

import numpy as np
import pybullet as p
import trimesh
from scipy.spatial.transform import Rotation

import igibson
from igibson.objects.articulated_object import URDFObject
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils import utils


def main(selection="user", headless=False, short_exec=False):
    """
    Shows how to obtain the bounding box of an articulated object.
    Draws the bounding box around the loaded object, a cabinet, while it moves.
    Visible only in the pybullet GUI.
    """
    print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    # s = Simulator(mode="headless", use_pb_gui=True)
    s = Simulator(gravity=0,
                  image_width=720, image_height=720,
                  use_pb_gui=True)
    scene = EmptyScene(floor_plane_rgba=[0.6, 0.6, 0.6, 1])
    s.import_scene(scene)

    # Banana is a single-link object and Door is a multi-link object.
    # banana_dir = os.path.join(igibson.ig_dataset_path, "objects/banana/09_0")
    # banana_filename = os.path.join(banana_dir, "09_0.urdf")
    # door_dir = os.path.join(igibson.ig_dataset_path, "objects/door/8930")
    # door_filename = os.path.join(door_dir, "8930.urdf")
    games_filename = '/home/ubuntu/Desktop/iGibson_study/iGibson/igibson/data/ig_dataset/objects/video_game/Pokémon_Omega_Ruby_Alpha_Sapphire_Dual_Pack_Nintendo_3DS/Pokémon_Omega_Ruby_Alpha_Sapphire_Dual_Pack_Nintendo_3DS.urdf'

    # banana = URDFObject(filename=banana_filename, category="banana", scale=np.array([3.0, 5.0, 2.0]),
    #                     merge_fixed_links=False)
    # door = URDFObject(filename=door_filename, category="door", scale=np.array([1.0, 2.0, 3.0]), merge_fixed_links=False)
    games = URDFObject(filename=games_filename,
                       category="games",
                       # scale=np.array([3.0, 5.0, 2.0]),
                       merge_fixed_links=False,
                       # bounding_box=True
                       )
    # s.import_object(banana)
    # s.import_object(door)
    s.import_object(games)
    # banana.set_position_orientation([2, 0, 0.75], [0, 0, 0, 0])
    # door.set_position_orientation([-2, 0, 2], Rotation.from_euler("XYZ", [0, 0, -np.pi / 4]).as_quat())
    games.set_position_orientation([1, 2, 3], Rotation.from_euler("XYZ", [np.pi / 4, np.pi / 4, np.pi / 4]).as_quat())

    # Main simulation loop
    try:
        steps = 0
        max_steps = -1 if not short_exec else 1000

        # Main recording loop
        while steps != max_steps:
            # Step simulation.
            s.step()

            line_idx = 0
            for obj in [games]:
                # Draw new debug lines for the bounding boxes.
                bbox_center, bbox_orn, bbox_bf_extent, bbox_wf_extent = obj.get_base_aligned_bounding_box(visual=True)
                bbox_frame_vertex_positions = np.array(list(itertools.product((1, -1), repeat=3))) * (
                        bbox_bf_extent / 2
                )
                bbox_transform = utils.quat_pos_to_mat(bbox_center, bbox_orn)
                world_frame_vertex_positions = trimesh.transformations.transform_points(
                    bbox_frame_vertex_positions, bbox_transform
                )
                print(world_frame_vertex_positions)

                aabb = p.getAABB(3)
                x1, y1, z1 = aabb[0]
                x2, y2, z2 = aabb[1]
                x0 = (x1 + x2) / 2
                y0 = (y1 + y2) / 2
                z0 = (z1 + z2) / 2
                a = abs(x1 - x2) / 2
                b = abs(y1 - y2) / 2
                c = abs(z1 - z2) / 2
                cube_bounding_box = [[x0 + a, y0 + b, z0 + c],
                                     [x0 + a, y0 - b, z0 + c],
                                     [x0 + a, y0 + b, z0 - c],
                                     [x0 + a, y0 - b, z0 - c],
                                     [x0 - a, y0 + b, z0 + c],
                                     [x0 - a, y0 - b, z0 + c],
                                     [x0 - a, y0 + b, z0 - c],
                                     [x0 - a, y0 - b, z0 - c]]
                print(cube_bounding_box)

                print('----------------------------------------------------------')

                for i, from_vertex in enumerate(cube_bounding_box):
                    for j, to_vertex in enumerate(cube_bounding_box):
                        if j <= i:
                            ret_val = p.addUserDebugLine(
                                from_vertex,
                                to_vertex,
                                lineColorRGB=[1.0, 0.0, 0.0],
                                lineWidth=1,
                                lifeTime=0,
                                replaceItemUniqueId=-1 if steps == 0 else line_idx,
                            )
                            if not headless:
                                assert ret_val == line_idx
                            line_idx += 1

            steps += 1
    finally:
        s.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
