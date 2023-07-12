import argparse
import sys

import numpy as np
import trimesh

parser = argparse.ArgumentParser("mesh info extractor")
parser.add_argument("filename", type=str)
args = parser.parse_args()

filename = args.filename

mesh = trimesh.load(filename)
print("Loaded mesh from", filename, "=", mesh)

to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
to_origin = np.linalg.inv(to_origin)
x, y, z = to_origin[:3, 3]
print("pos =", (x, y, z))
# Trimesh does things in the "wrong" order
# w, x, y, z = trimesh.transformations.quaternion_from_matrix(to_origin)
# orn = [x, y, z, w]
orn = trimesh.transformations.euler_from_matrix(to_origin)
print("orn =", orn)
print("extents =", extents)


template = """
    <collision>
      <origin rpy="%f %f %f" xyz="%f %f %f"/>
      <geometry>
        <box size="%f %f %f"/>
      </geometry>
    </collision>"""

print()
print()
print(template % (orn[0], orn[1], orn[2], x, y, z, extents[0], extents[1], extents[2]))
