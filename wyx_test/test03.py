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
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.render.profiler import Profiler
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene


def main(selection="user", headless=False, short_exec=False):
    scene_id = 'Beechwood_0_int'
    settings = MeshRendererSettings(enable_shadow=True, msaa=True, blend_highlight=True, )
    # if platform == "darwin":
    #     settings.texture_scale = 0.5
    s = Simulator(
        gravity=9.8,
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

    # s = Simulator(mode="headless", use_pb_gui=True)
    # s = Simulator(gravity=9.8,
    #               image_width=720, image_height=720,
    #               use_pb_gui=False)
    # scene = EmptyScene(floor_plane_rgba=[0.6, 0.6, 0.6, 1])
    s.import_scene(scene)

    shapenet_filename = '/remote-home/2132917/Desktop/ShapeNet_dataset/ShapeNetCore.v2/03001627/1a6f615e8b1b5ae4dbbc9440457e303e/models/model_normalized.obj'
    shapenet = ShapeNetObject(path=shapenet_filename, position=[0, 0, 1])

    s.import_object(shapenet)

    # Main simulation loop
    try:
        # Main recording loop
        while True:
            # Step simulation.
            s.step()


    finally:
        s.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
