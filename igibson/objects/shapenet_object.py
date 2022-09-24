import numpy as np
import pybullet as p

from igibson.objects.stateful_object import StatefulObject


class ShapeNetObject(StatefulObject):
    """
    ShapeNet object
    Reference: https://www.shapenet.org/
    """

    def __init__(self, path, scale=1.0, position=[0, 0, 0], orientation=[0, 0, 0], **kwargs):
        super(ShapeNetObject, self).__init__(**kwargs)
        self.filename = path
        self.scale = scale
        self.position = position
        self.orientation = orientation

        self._default_mass = 3.0
        self._default_transform = {
            "position": [0, 0, 0],
            "orientation_quat": [1.0 / np.sqrt(2), 0, 0, 1.0 / np.sqrt(2)],
        }
        pose = p.multiplyTransforms(
            positionA=self.position,
            orientationA=p.getQuaternionFromEuler(self.orientation),
            positionB=self._default_transform["position"],
            orientationB=self._default_transform["orientation_quat"],
        )
        self.pose = {
            "position": pose[0],
            "orientation_quat": pose[1],
        }

        self.init_center_point_bounding_box()

    def _load(self, simulator):
        """
        Load the object into pybullet
        """
        collision_id = p.createCollisionShape(p.GEOM_MESH, fileName=self.filename, meshScale=self.scale)
        body_id = p.createMultiBody(
            basePosition=self.pose["position"],
            baseOrientation=self.pose["orientation_quat"],
            baseMass=self._default_mass,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=-1,
        )
        simulator.load_object_in_renderer(self, body_id, self.class_id, **self._rendering_params)

        return [body_id]

    def init_center_point_bounding_box(self):
        xmin, xmax, ymin, ymax, zmin, zmax = 1000, -1000, 1000, -1000, 1000, -1000

        points = []
        with open(self.filename) as file:
            while True:
                line = file.readline()

                if not line:
                    break
                strs = line.split(" ")  # 读取到的数据
                if strs[0] == "v":
                    points.append([float(strs[1]), float(strs[2]), float(strs[3])])

        for point in points:
            if point[0] < xmin:
                xmin = point[0]
            if point[0] > xmax:
                xmax = point[0]
            if point[1] < ymin:
                ymin = point[1]
            if point[1] > ymax:
                ymax = point[1]
            if point[2] < zmin:
                zmin = point[2]
            if point[2] > zmax:
                zmax = point[2]

        x0 = (xmin + xmax) / 2
        y0 = (ymin + ymax) / 2
        z0 = (zmin + zmax) / 2
        a = abs(xmin - xmax) / 2
        b = abs(ymin - ymax) / 2
        c = abs(zmin - zmax) / 2
        self.init_center = [x0, y0, z0]
        self.init_bounding_box = [[x0 + a, y0 + b, z0 + c],
                                  [x0 + a, y0 - b, z0 + c],
                                  [x0 + a, y0 + b, z0 - c],
                                  [x0 + a, y0 - b, z0 - c],
                                  [x0 - a, y0 + b, z0 + c],
                                  [x0 - a, y0 - b, z0 + c],
                                  [x0 - a, y0 + b, z0 - c],
                                  [x0 - a, y0 - b, z0 - c]]

        self.update_center_point_bounding_box()

    def update_center_point_bounding_box(self):
        r = p.getMatrixFromQuaternion(self.pose['orientation_quat'])
        r = np.array([[r[0], r[1], r[2]],
                      [r[3], r[4], r[5]],
                      [r[6], r[7], r[8]]])
        t = np.array([[self.pose['position'][0]], [self.pose['position'][1]], [self.pose['position'][2]]])

        self.bounding_box = self.init_bounding_box.copy()  # 物体的立体框，8个顶点
        for i in range(8):
            point = np.array([[self.init_bounding_box[i][0]],
                              [self.init_bounding_box[i][1]],
                              [self.init_bounding_box[i][2]]])
            res = np.matmul(r, point) + t

            self.bounding_box[i] = [res[0][0], res[1][0], res[2][0]]

        self.center = np.matmul(r, np.array([[self.init_center[0]], [self.init_center[1]], [self.init_center[0]]])) + t

        # 标记物体坐标系的xyz轴
        object_coordinate_x = np.matmul(r, np.array(
            [[self.init_center[0] + 1], [self.init_center[1]], [self.init_center[0]]])) + t
        object_coordinate_y = np.matmul(r, np.array(
            [[self.init_center[0]], [self.init_center[1] + 1], [self.init_center[0]]])) + t
        object_coordinate_z = np.matmul(r, np.array(
            [[self.init_center[0]], [self.init_center[1]], [self.init_center[0] + 1]])) + t
        self.object_coordinate = [object_coordinate_x, object_coordinate_y, object_coordinate_z]

        return self.center, self.bounding_box, self.object_coordinate
