from data.parse_openmx import read_openmx39
from tight_binding_model import create_TBModel
from copy import deepcopy


def calcassistvars(Total_NumOrbs):
    # generate accumulated-indices
    numorb_base = [0] * len(Total_NumOrbs)
    numorb_base[0] = 0
    for i in range(1, len(Total_NumOrbs)):
        numorb_base[i] = numorb_base[i - 1] + Total_NumOrbs[i - 1]
    return numorb_base


# def construct_tight_binding_model_from_openmx:
result_dict = read_openmx39("data/GaAs.openmx39")
atomnum = result_dict["atomnum"]
SpinP_switch = result_dict["SpinP_switch"]
atv = result_dict["atv"]
atv_ijk = result_dict["atv_ijk"]
Total_NumOrbs = result_dict["Total_NumOrbs"]
FNAN = result_dict["FNAN"]
natn = result_dict["natn"]
ncn = result_dict["ncn"]
tv = result_dict["tv"]
Hk = result_dict["Hk"]
iHk = result_dict["iHk"]
OLP = result_dict["OLP"]
OLP_r = result_dict["OLP_r"]

numorb_base = calcassistvars(Total_NumOrbs)
# print("numorb_base", numorb_base)
Total_NumOrbs_sum = sum(Total_NumOrbs)
atv = atv * 0.529177249
tv = tv * 0.529177249

atv = None
# def element_multiply(ll):
#     return list(map(lambda _: _ * 27.211399, ll))
tmp = deepcopy(Hk)
# for t in [Hk, iHk]:
#     if t is not None:
#         t = list(
#             map(lambda strip_1: list(map(lambda strip_2: list(map(lambda nd_array: nd_array * 27.21139, strip_2)), strip_1)),
#                 t))
#         # t *= 27.211399  # Hartree to eV
if Hk is not None:
    Hk = list(
        map(lambda strip_1: list(
            map(lambda strip_2: list(map(lambda nd_array: nd_array * 27.211399, strip_2)), strip_1)),
            Hk))
if iHk is not None:
    iHk = list(
        map(lambda strip_1: list(
            map(lambda strip_2: list(map(lambda nd_array: nd_array * 27.211399, strip_2)), strip_1)),
            iHk))
if OLP_r is not None:
    OLP_r = list(
        map(lambda strip_1: list(
            map(lambda strip_2: list(map(lambda nd_array: nd_array * 0.529177249, strip_2)), strip_1)),
            OLP_r))

# OLP_r = OLP_r * 0.529177249
nm = create_TBModel(Total_NumOrbs_sum, tv, isorthogonal=False)
if SpinP_switch == 0:
    for i in range(atomnum):
        for j in range(FNAN[0]):
            for ii in range(Total_NumOrbs[i]):
                for jj in range(Total_NumOrbs[natn[i][j] - 1]):
                    # print("i,j,ii,jj", i, j, ii, jj)
                    # print("Hk[0][i][j][jj, ii]", Hk[0][i][j][jj, ii])
                    # print("natnn[i][j]", natn[i][j])
                    # print("atv_ijk[:, ncn[i][j]]", atv_ijk[:, ncn[i][j]])
                    # print("numorb_base[i]", numorb_base[i])
                    # print("numorb_base[natn[i][j]]", numorb_base[natn[i][j]])
                    # print("\n")
                    nm.set_hopping(tuple(atv_ijk[:, ncn[i][j]]), numorb_base[i] + ii, numorb_base[natn[i][j] - 1] + jj,
                                   Hk[0][i][j][jj, ii])
                    nm.set_overlap(tuple(atv_ijk[:, ncn[i][j]]), numorb_base[i] + ii, numorb_base[natn[i][j] - 1] + jj,
                                   OLP[i][j][jj, ii])

    for i in range(atomnum):
        for j in range(FNAN[0]):
            for ii in range(Total_NumOrbs[i]):
                for jj in range(Total_NumOrbs[natn[i][j] - 1]):
                    for alpha in range(1, 4):
                        nm.set_position(tuple(atv_ijk[:, ncn[i][j]]), numorb_base[i] + ii,
                                        numorb_base[natn[i][j] - 1] + jj, alpha,
                                        OLP_r[alpha-1][i][j][jj, ii])
# print(nm.hoppings)


# # if __name__ == '__main__':
#     construct_tight_binding_model_from_openmx()
