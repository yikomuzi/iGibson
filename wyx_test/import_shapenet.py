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


def main(selection="user", headless=False, short_exec=False):
    # s = Simulator(mode="headless", use_pb_gui=True)
    s = Simulator(gravity=9.8,
                  image_width=720, image_height=720,
                  use_pb_gui=False)
    scene = EmptyScene(floor_plane_rgba=[0.6, 0.6, 0.6, 1])
    s.import_scene(scene)

    shapenet_filename = '/remote-home/2132917/Desktop/ShapeNet_dataset/ShapeNetCore.v2/03001627/1a6f615e8b1b5ae4dbbc9440457e303e/models/model_normalized.obj'
    shapenet = ShapeNetObject(path=shapenet_filename, position=[1, 2, 3])

    s.import_object(shapenet)

    # Main simulation loop
    try:
        steps = 0
        max_steps = -1 if not short_exec else 1000

        # Main recording loop
        while steps != max_steps:
            # Step simulation.
            s.step()
            shapenet.update_center_point_bounding_box()
            print(shapenet.get_poses())

            for i, from_vertex in enumerate(shapenet.bounding_box):
                for j, to_vertex in enumerate(shapenet.bounding_box):
                    if j <= i:
                        p.addUserDebugLine(
                            from_vertex,
                            to_vertex,
                            lineColorRGB=[1.0, 0.0, 0.0],
                            lineWidth=1,
                            lifeTime=0,
                        )

            steps += 1
    finally:
        s.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
