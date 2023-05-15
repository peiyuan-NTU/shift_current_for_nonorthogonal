import numpy as np


class UniformMesh:
    def __init__(self, mesh_size, startpoint, region_size, dimension, end_boundary, meshsize_cumprod):
        self.mesh_size = mesh_size
        self.startpoint = startpoint
        self.region_size = region_size
        self.dimension = dimension
        self.end_boundary = end_boundary
        self.meshsize_cumprod = meshsize_cumprod
        # self.x_step = 1 / mesh_size[0]
        # self.y_step = 1 / mesh_size[1]
        # self.z_step = 1 / mesh_size[2]

    def __iter__(self):
        self.a = np.zeros(3, dtype=int)
        return self

    def __next__(self):
        # if int(self.a[0] * self.mesh_size[0]) == 1 and int(self.a[1] * self.mesh_size[1]) == 1 and int(
        #         self.a[2] * self.mesh_size[2]) == 1:
        #     raise StopIteration
        # print("a", self.a)
        if self.a[0] + 1 == self.mesh_size[0] and self.a[1] + 1 == self.mesh_size[1] and self.a[2] + 1 == \
                self.mesh_size[2]:
            raise StopIteration
        else:
            self.a[0] += 1
            # print("first", self.a)
            if self.a[0] == self.mesh_size[0]:
                self.a[0] = 0
                self.a[1] += 1
                # print("second", self.a)
                if self.a[1] == self.mesh_size[1]:
                    self.a[1] = 0
                    self.a[2] += 1
                    # print("third", self.a)
                    if self.a[2] == self.mesh_size[2]:
                        raise StopIteration
            point = self.a / self.mesh_size
        # print("return", self.a, "\n")
        return point

        # self.a[0] += self.x_step
        # if self.a[0] >= self.region_size[0]:
        #     self.a[0] = 0
        #     self.a[1] += self.y_step
        #     if self.a[1] >= self.region_size[1]:
        #         self.a[1] = 0
        #         self.a[2] += self.z_step
        #         if self.a[2] >= self.region_size[2]:
        #             self.a[2] = 1

        # if self.a[0] * self.mesh_size[0] >= 1:
        #     self.a[0] = 0
        #     self.a[1] += self.y_step
        #     if self.a[1] * self.mesh_size[1] >= 1:
        #         self.a[1] = 0
        #         self.a[2] += self.z_step
        #         if self.a[2] * self.mesh_size[2] >= 1:
        #             self.a[2] = 0


def create_uniform_mesh(mesh_size, startpoint=None, region_size=None, endboundary=False):
    if startpoint is None:
        startpoint = np.zeros(len(mesh_size))
    if region_size is None:
        region_size = np.ones(len(mesh_size))

    assert len(mesh_size) == len(startpoint) == len(region_size), "Arguments are of incompatible shapes."
    assert all(x > 0 for x in mesh_size), "meshsize should contain positive integers."
    assert all(x > 0 for x in region_size), "regionsize should contain positive numbers."

    return UniformMesh(mesh_size, startpoint, region_size, len(mesh_size), endboundary, np.cumprod(mesh_size))


if __name__ == '__main__':
    mesh = create_uniform_mesh([100, 100, 1])
    for i in mesh:
        print(i)
        # pass
