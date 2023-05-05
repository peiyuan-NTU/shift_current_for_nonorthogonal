import numpy as np
import os
import re


def multiread(T, f, size):
    ret = np.fromfile(f, dtype=T, count=size)
    return ret


def read_mixed_matrix(dtype, f, dims):
    ret = []
    for i in dims:
        t = np.fromfile(f, dtype=dtype, count=i)
        ret.append(t)
    return ret


def read_matrix_in_mixed_matrix(dtype, f, spins, atomnum, FNAN, natn, Total_NumOrbs):
    ret = [[[np.zeros((Total_NumOrbs[natn[ai][aj_inner] - 1], Total_NumOrbs[ai]), dtype=dtype) for aj_inner in
             range(FNAN[ai])] for ai in range(atomnum)] for _ in range(spins)]
    for spin in range(spins):
        for ai in range(atomnum):
            for aj_inner in range(FNAN[ai]):
                t = np.fromfile(f, dtype=dtype,
                                count=Total_NumOrbs[natn[ai][aj_inner] - 1] * Total_NumOrbs[ai]).reshape(
                    (Total_NumOrbs[natn[ai][aj_inner] - 1], Total_NumOrbs[ai]))
                ret[spin][ai][aj_inner] = t
    return ret


def read_3d_vecs(dtype, f, num):
    ret = np.fromfile(f, dtype=dtype, count=4 * num).reshape((num, 4)).T[1:4]
    return ret


def read_openmx39(file_name):
    with open(file_name, "rb") as f:
        atomnum, SpinP_switch, Catomnum, Latomnum, Ratomnum, TCpyCell, order_max = multiread(np.int32, f, 7)
        # print(atomnum, SpinP_switch, Catomnum, Latomnum, Ratomnum, TCpyCell, order_max)
        assert (SpinP_switch >> 2) == 3
        SpinP_switch &= 0x03
        # print("spin polarized", SpinP_switch)
        atv, atv_ijk = [read_3d_vecs(np.float64 if i == 0 else np.int32, f, TCpyCell + 1) for i in range(2)]
        Total_NumOrbs, FNAN = [multiread(np.int32, f, atomnum) for _ in range(2)]
        FNAN = [x + 1 for x in FNAN]
        natn = read_mixed_matrix(np.int32, f, FNAN)
        ncn = [[i + 1 for i in x] for x in read_mixed_matrix(np.int32, f, FNAN)]
        # tv= read_3d_vecs(np.float64, f, 3)
        tv, rtv, Gxyz = [read_3d_vecs(np.float64, f, _) for _ in [3, 3, atomnum]]

        Hk = read_matrix_in_mixed_matrix(np.float64, f, SpinP_switch + 1, atomnum, FNAN, natn, Total_NumOrbs)
        Hk = list(
            map(lambda strip_1: list(
                map(lambda strip_2: list(map(lambda nd_array: nd_array.T, strip_2)), strip_1)),
                Hk))


        iHk = read_matrix_in_mixed_matrix(np.float64, f, 3, atomnum, FNAN, natn,
                                          Total_NumOrbs) if SpinP_switch == 3 else None
        if iHk is not None:
            iHk = list(
                map(lambda strip_1: list(
                    map(lambda strip_2: list(map(lambda nd_array: nd_array.T, strip_2)), strip_1)),
                    iHk))
        OLP = read_matrix_in_mixed_matrix(np.float64, f, 1, atomnum, FNAN, natn, Total_NumOrbs)[0]
        OLP = list(map(lambda strip_1: list(map(lambda nd_array: nd_array.T, strip_1)), OLP))



        OLP_r = []
        for i in range(3):
            for order in range(order_max):
                # print("i", i, "order", order)
                tmp = read_matrix_in_mixed_matrix(np.float64, f, 1, atomnum, FNAN, natn, Total_NumOrbs)[0]
                if order == 0:
                    OLP_r.append(tmp)
        OLP_r = list(
                map(lambda strip_1: list(
                    map(lambda strip_2: list(map(lambda nd_array: nd_array.T, strip_2)), strip_1)),
                    OLP_r))
        OLP_p = read_matrix_in_mixed_matrix(np.float64, f, 3, atomnum, FNAN, natn, Total_NumOrbs)
        # OLP_p = list(map(lambda strip_1: list(map(lambda nd_array: nd_array.T, strip_1)), OLP_p))
        DM = read_matrix_in_mixed_matrix(np.float64, f, SpinP_switch + 1, atomnum, FNAN, natn, Total_NumOrbs)
        # DM = list(map(lambda strip_1: list(map(lambda nd_array: nd_array.T, strip_1)), DM))
        iDM = read_matrix_in_mixed_matrix(np.float64, f, 2, atomnum, FNAN, natn, Total_NumOrbs)
        # iDM = list(map(lambda strip_1: list(map(lambda nd_array: nd_array.T, strip_1)), iDM))
        solver = multiread(np.int32, f, 1)[0]
        chem_p, E_temp = multiread(np.float64, f, 2)
        dipole_moment_core, dipole_moment_background = [multiread(np.float64, f, 3) for _ in range(2)]
        Valence_Electrons, Total_SpinS = multiread(np.float64, f, 2)
        dummy_blocks = multiread(np.int32, f, 1)[0]

        for i in range(dummy_blocks):
            multiread(np.uint8, f, 256)

        def strip1(s):
            # print(s)
            startpos = 0
            for i in range(len(s) + 1):
                # print(len(s), i, s[i] & 0x80, str.isspace(chr(s[i] & 0x7f)))
                if i + 1 == len(s) or s[i] & 0x80 != 0 or not str.isspace(chr(s[i] & 0x7f)):
                    startpos = i
                    break
            return s[startpos:]

        def startswith1(s: bytearray, prefix: bytearray) -> bool:
            return len(s) >= len(prefix) and s[:len(prefix)] == prefix

        target_line = bytearray("Fractional coordinates of the final structure", "utf-8")
        next_line = bytearray(f.readline())

        # while not startswith1(strip1(next_line), target_line):
        while 1:
            stripped = strip1(next_line)
            if startswith1(stripped, target_line):
                break
            if f.tell() == os.fstat(f.fileno()).st_size:
                raise Exception(
                    "Atom positions not found. Please check if the .out file was appended to the end of .scfout file!")
            next_line = bytearray(f.readline())

        for i in range(2):
            assert f.readline() == b'***********************************************************\n'
        assert f.readline() == b"\n"
        #
        #
        #
        atom_frac_pos = np.zeros((3, atomnum))
        for i in range(atomnum):
            line = f.readline()

            m = re.match(r"^\s*\d+\s+\w+\s+([0-9+-.Ee]+)\s+([0-9+-.Ee]+)\s+([0-9+-.Ee]+)", line.decode("utf-8"))
            # print(line)
            # print(m.groups())
            atom_frac_pos[:, i] = np.array(list(map(float, m.groups())))
        atom_pos = tv.dot(atom_frac_pos)

        f.close()
        #
        # # # use the atom_pos to fix
        for axis in range(3):
            for i in range(atomnum):
                for j in range(FNAN[i]):
                    OLP_r[axis][i][j] += atom_pos[axis, i] * OLP[i][j]
        # #
        # # # fix type mismatch
        atv_ijk = atv_ijk.astype(np.int16)

        result_dict = {"atomnum": atomnum,
                       "SpinP_switch": SpinP_switch,
                       "atv": atv,
                       "atv_ijk": atv_ijk,
                       "Total_NumOrbs": Total_NumOrbs,
                       "FNAN": FNAN,
                       "natn": natn,
                       "ncn": ncn,
                       "tv": tv,
                       "Hk": Hk,
                       "iHk": iHk,
                       "OLP": OLP,
                       "OLP_r": OLP_r}
        return result_dict


# if __name__ == '__main__':
#     file_path = "GaAs.openmx39"
#     with open(file_path, "rb") as f:
#         atomnum, SpinP_switch, Catomnum, Latomnum, Ratomnum, TCpyCell, order_max = multiread(np.int32, f, 7)
#         # print(atomnum, SpinP_switch, Catomnum, Latomnum, Ratomnum, TCpyCell, order_max)
#         assert (SpinP_switch >> 2) == 3
#         SpinP_switch &= 0x03
#         # print("spin polarized", SpinP_switch)
#         atv, atv_ijk = [read_3d_vecs(np.float64 if i == 0 else np.int32, f, TCpyCell + 1) for i in range(2)]
#         Total_NumOrbs, FNAN = [multiread(np.int32, f, atomnum) for _ in range(2)]
#         FNAN = [x + 1 for x in FNAN]
#         natn = read_mixed_matrix(np.int32, f, FNAN)
#         ncn = [[i + 1 for i in x] for x in read_mixed_matrix(np.int32, f, FNAN)]
#         # tv= read_3d_vecs(np.float64, f, 3)
#         tv, rtv, Gxyz = [read_3d_vecs(np.float64, f, _) for _ in [3, 3, atomnum]]
#
#         Hk = read_matrix_in_mixed_matrix(np.float64, f, SpinP_switch + 1, atomnum, FNAN, natn, Total_NumOrbs)
#         iHk = read_matrix_in_mixed_matrix(np.float64, f, 3, atomnum, FNAN, natn,
#                                           Total_NumOrbs) if SpinP_switch == 3 else None
#         OLP = read_matrix_in_mixed_matrix(np.float64, f, 1, atomnum, FNAN, natn, Total_NumOrbs)[0]
#         # tmp = read_matrix_in_mixed_matrix(np.float64, f, 1, atomnum, FNAN, natn, Total_NumOrbs)[0]
#         # tmp = read_matrix_in_mixed_matrix(np.float64, f, 1, atomnum, FNAN, natn, Total_NumOrbs)[0]
#         # tmp = read_matrix_in_mixed_matrix(np.float64, f, 1, atomnum, FNAN, natn, Total_NumOrbs)[0]
#         OLP_r = []
#         for i in range(3):
#             for order in range(order_max):
#                 # print("i", i, "order", order)
#                 tmp = read_matrix_in_mixed_matrix(np.float64, f, 1, atomnum, FNAN, natn, Total_NumOrbs)[0]
#                 if order == 0:
#                     OLP_r.append(tmp)
#         OLP_p = read_matrix_in_mixed_matrix(np.float64, f, 3, atomnum, FNAN, natn, Total_NumOrbs)
#         DM = read_matrix_in_mixed_matrix(np.float64, f, SpinP_switch + 1, atomnum, FNAN, natn, Total_NumOrbs)
#         iDM = read_matrix_in_mixed_matrix(np.float64, f, 2, atomnum, FNAN, natn, Total_NumOrbs)
#         solver = multiread(np.int32, f, 1)[0]
#         chem_p, E_temp = multiread(np.float64, f, 2)
#         dipole_moment_core, dipole_moment_background = [multiread(np.float64, f, 3) for _ in range(2)]
#         Valence_Electrons, Total_SpinS = multiread(np.float64, f, 2)
#         dummy_blocks = multiread(np.int32, f, 1)[0]
#
#         for i in range(dummy_blocks):
#             multiread(np.uint8, f, 256)
#
#
#         def strip1(s):
#             # print(s)
#             startpos = 0
#             for i in range(len(s) + 1):
#                 # print(len(s), i, s[i] & 0x80, str.isspace(chr(s[i] & 0x7f)))
#                 if i + 1 == len(s) or s[i] & 0x80 != 0 or not str.isspace(chr(s[i] & 0x7f)):
#                     startpos = i
#                     break
#             return s[startpos:]
#
#
#         def startswith1(s: bytearray, prefix: bytearray) -> bool:
#             return len(s) >= len(prefix) and s[:len(prefix)] == prefix
#
#
#         target_line = bytearray("Fractional coordinates of the final structure", "utf-8")
#         next_line = bytearray(f.readline())
#         found = False
#         # while not startswith1(strip1(next_line), target_line):
#         while 1:
#             stripped = strip1(next_line)
#             if startswith1(stripped, target_line):
#                 found = True
#                 break
#             if f.tell() == os.fstat(f.fileno()).st_size:
#                 raise Exception(
#                     "Atom positions not found. Please check if the .out file was appended to the end of .scfout file!")
#             next_line = bytearray(f.readline())
#         # print("found", next_line)
#         # print(f.readline())
#         # print(f.readline())
#         for i in range(2):
#             assert f.readline() == b'***********************************************************\n'
#         assert f.readline() == b"\n"
#         #
#         #
#         #
#         atom_frac_pos = np.zeros((3, atomnum))
#         for i in range(atomnum):
#             line = f.readline()
#
#             m = re.match(r"^\s*\d+\s+\w+\s+([0-9+-.Ee]+)\s+([0-9+-.Ee]+)\s+([0-9+-.Ee]+)", line.decode("utf-8"))
#             print(line)
#             print(m.groups())
#             atom_frac_pos[:, i] = np.array(list(map(float, m.groups())))
#         atom_pos = tv.dot(atom_frac_pos)
#
#         f.close()
#         #
#         # # # use the atom_pos to fix
#         for axis in range(3):
#             for i in range(atomnum):
#                 for j in range(FNAN[i]):
#                     OLP_r[axis][i][j] += atom_pos[axis, i] * OLP[i][j]
#         # #
#         # # # fix type mismatch
#         atv_ijk = atv_ijk.astype(np.int16)


if __name__ == '__main__':
    file_path = "GaAs.openmx39"
    result = read_openmx39(file_path)
