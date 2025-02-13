from __future__ import annotations

from math import floor
from pathlib import Path

import numpy as np


class DriftTimes:
    def __init__(self, r, z, t) -> None:
        self.r = r
        self.z = z
        self.t = t


class ReadDTFile:
    def __init__(self, filename, x, y, z) -> None:
        datatype = filename.split("/")[-1].split(".")[-1]
        self.r = []
        self.z = []
        self.t = []
        self.xpos = x
        self.ypos = y
        self.zpos = z

        self.drift_times = []
        self.filename = filename

        with Path.open(filename) as file:
            fl = file.readlines()

            for line in fl:
                line_list = line.split(" ") if datatype == "dat" else line.split(",")

                r = float(line_list[0])
                z = float(line_list[1])
                t = float(line_list[2].strip())

                self.r.append(r)
                self.z.append(z)
                self.t.append(t)

                self.drift_times.append(DriftTimes(r, z, t))

        rs, zs = [], []
        for xi, zi in zip(self.r, self.z):
            if xi not in rs:
                rs.append(xi)
            if zi not in zs:
                zs.append(zi)
        self.dz = abs(zs[-1] - zs[-2])
        self.dr = abs(rs[-1] - rs[-2])
        self.maxz = max(zs)
        self.maxr = max(rs)
        self.minz = min(zs)
        self.minr = min(rs)

    def GetIndex(self, r, z):
        rGrid = floor(r / self.dr + 0.5) * self.dr
        zGrid = floor(z / self.dz + 0.5) * self.dz
        numZ = int((self.maxz - self.minz) / self.dz + 1)
        i = int((rGrid - self.minr) / self.dr * numZ + (zGrid - self.minz) / self.dz)
        nDT = len(self.drift_times)

        if i >= nDT:
            return nDT - 1
        return i

    def GetTime(self, x, y, input_z):
        r = self.get_r(x, y)
        z = input_z - self.zpos

        r1 = floor(r / self.dr) * self.dr
        z1 = floor(z / self.dz) * self.dz

        t11 = self.drift_times[self.GetIndex(r1, z1)].t
        if r1 == r and z1 == z:
            return t11

        # Bilinear interpolation. Might do cubic interpolation eventually.
        r2 = r1 + self.dr
        z2 = z1 + self.dz

        t12 = self.drift_times[self.GetIndex(r1, z2)].t
        t21 = self.drift_times[self.GetIndex(r2, z1)].t
        t22 = self.drift_times[self.GetIndex(r2, z2)].t

        return (
            1
            / ((r2 - r1) * (z2 - z1))
            * (
                t11 * (r2 - r) * (z2 - z)
                + t21 * (r - r1) * (z2 - z)
                + t12 * (r2 - r) * (z - z1)
                + t22 * (r - r1) * (z - z1)
            )
        )

    def get_r(self, x, y):
        return np.sqrt((x - self.xpos) ** 2 + (y - self.ypos) ** 2)
