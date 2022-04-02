import os
import sys
import numpy as np
from PIL import Image
from .color_palette import ColorPalette
from .imgprocessing import rgb_threshold
from .files import purename
from .utils import load_variables
from .plotter import Xls_Color_Spectrum

#   对一张图像的颜色进行聚类分析（最多指定5类）Refer to https://github.com/Dilan1020/PyColorPalette
#   background_threshold：tuple，可以直接对图像进行阈值过滤，背景不会纳入到聚类算法中
#       对于普通的WSI，推荐(1,-220,-220)，表示R超过1（不会是黑色），G、B小于220（滤除白色）的像素才会被考虑进来
def color_palette(image_path, n=5, foreground=None):
    pal = ColorPalette(image_path, n, foreground=foreground)
    clustering_result = pal.get_top_colors(n=n, ratio=True, rounded=False)

    weights_list = []
    means_list = []
    for item in clustering_result:
        means_list.append(list(item[0]))
        weights_list.append(item[1])
    weights = np.array(weights_list)
    weights = weights / np.sum(weights)
    means = np.array(means_list)

    spectrum = {}
    spectrum['file_name'] = purename(image_path) if not isinstance(image_path, np.ndarray) else 'None'
    spectrum['type'] = 'palette'
    spectrum['weights'] = weights
    spectrum['means'] = means
    spectrum['covariances'] = np.nan
    spectrum['object'] = clustering_result
    return spectrum


def color_gmm(image_path, n=5, foreground=None, max_point=100000):
    pixels = np.array(Image.open(image_path).convert('RGB'))

    #   新增加的代码，用于处理背景
    if foreground is not None:
        with_foreground = True

        if isinstance(foreground, tuple) or isinstance(foreground, list):
            valid_map = rgb_threshold(np.array(pixels),
                                      r_th=foreground[0],
                                      g_th=foreground[1],
                                      b_th=foreground[2], )
        else:
            valid_map = foreground
        valid_map = valid_map.reshape((-1))
    else:
        with_foreground = False
        valid_map = None

    #   数据重排
    pixels = pixels.reshape((-1, 3))
    if with_foreground:
        pixels = pixels[valid_map]

    if len(pixels) > max_point:
        idx = np.random.permutation(len(pixels))
        pixels = pixels[idx[0:max_point]]

    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=n, covariance_type='full')
    gmm.fit(pixels)

    spectrum = {}
    spectrum['file_name'] = purename(image_path)
    spectrum['type'] = 'gmm'
    spectrum['weights'] = gmm.weights_
    spectrum['means'] = gmm.means_
    spectrum['covariances'] = gmm.covariances_
    spectrum['object'] = gmm

    return spectrum


#   JS Distance between two GMMs, borrowed from https://stackoverflow.com/questions/26079881/kl-divergence-of-two-gmms
def gmm_js(gmm_p, gmm_q, n_samples=10 ** 5):
    X, _ = gmm_p.sample(n_samples)
    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)
    log_mix_X = np.logaddexp(log_p_X, log_q_X)

    Y, _ = gmm_q.sample(n_samples)
    log_p_Y = gmm_p.score_samples(Y)
    log_q_Y = gmm_q.score_samples(Y)
    log_mix_Y = np.logaddexp(log_p_Y, log_q_Y)

    return (log_p_X.mean() - (log_mix_X.mean() - np.log(2))
            + log_q_Y.mean() - (log_mix_Y.mean() - np.log(2))) / 2


#   计算两个色彩谱的距离，色彩谱即如上的聚类结果，是由若干((R,G,B),P)组成的list
#   pixel_distance: 计算两个(R,G,B)tuple距离的函数，若为None则默认使用L2距离
def spectrum_distance(spectrum1, spectrum2, pixel_distance=None, method='assign'):
    #   检查计算方法是否为支持的类型
    if method not in ['assign', 'js']:
        print('Unsupported method for calculating distance [assign|js]')
        raise NotImplementedError
    #   若为JS散度
    if method == 'js':
        if spectrum1['type'] == 'gmm' and spectrum2['type'] == 'gmm':
            return gmm_js(spectrum1['object'], spectrum2['object'])
        else:
            print('Spectrums involved are not gmm')
            raise ValueError
    #   若为cluster assignment
    elif method == 'assign':
        #   构建像素距离矩阵
        K1 = len(spectrum1['weights'])
        K2 = len(spectrum2['weights'])
        C = np.zeros((K1, K2))
        if pixel_distance is None:
            def pixel_distance(pixel1, pixel2):
                return np.sqrt(
                    (pixel1[0] - pixel2[0]) ** 2 + (pixel1[1] - pixel2[1]) ** 2 + (pixel1[2] - pixel2[2]) ** 2)
        for k1 in range(K1):
            for k2 in range(K2):
                C[k1, k2] = pixel_distance(spectrum1['means'][k1, :], spectrum2['means'][k2, :])

        #   解线性规划问题
        def _get_row(row):
            #   获取代表某行为1，其他为0的指示向量
            zero_mat = np.zeros((K1, K2))
            zero_mat[row, :] = 1
            return zero_mat.reshape(int(K1 * K2), order="C")

        def _get_col(col):
            #   获取代表某列为1，其他为0的指示向量
            zero_mat = np.zeros((K1, K2))
            zero_mat[:, col] = 1
            return zero_mat.reshape(int(K1 * K2), order="C")

        P1 = spectrum1['weights']
        P2 = spectrum2['weights']

        P1 = P1 / np.sum(P1)  # 归一化
        P2 = P2 / np.sum(P2)

        #   如果spectrum1只有一个簇群，即A是一个行向量，那么直接按照P2分配即可
        if K1 == 1:
            return np.sum(P2 * np.reshape(C, -1))
        #   如果spectrum2只有一个簇群，即A是一个列向量，那么直接按照P1分配即可
        if K2 == 1:
            return np.sum(P1 * np.reshape(C, -1))

        C_vector = C.reshape(int(K1 * K2), order="C")

        #   A[i,j]表示把谱1的第i簇指派给谱2的第j簇，输入约束是A的列和等于谱B的比例
        input_constrain = np.array([_get_col(col) for col in range(K2)])
        output_constrain = np.array([_get_row(row) for row in range(K1)])
        A_eq = np.concatenate([input_constrain, output_constrain], axis=0)
        b_eq = np.concatenate([P2, P1])

        from scipy import optimize as op
        res = op.linprog(c=C_vector,
                         A_eq=A_eq,
                         b_eq=b_eq)
        return res['fun']

        # #   --------------原先的版本--------------------
        # #   构建像素距离矩阵
        # K1 = len(spectrum1)
        # K2 = len(spectrum2)
        # C = np.zeros((K1, K2))
        # if pixel_distance is None:
        #     def pixel_distance(pixel1, pixel2):
        #         return np.sqrt(
        #             (pixel1[0] - pixel2[0]) ** 2 + (pixel1[1] - pixel2[1]) ** 2 + (pixel1[2] - pixel2[2]) ** 2)
        # for k1 in range(K1):
        #     for k2 in range(K2):
        #         C[k1, k2] = pixel_distance(spectrum1[k1][0], spectrum2[k2][0])
        #
        # #   解线性规划问题
        # def _get_row(row):
        #     #   获取代表某行为1，其他为0的指示向量
        #     zero_mat = np.zeros((K1, K2))
        #     zero_mat[row, :] = 1
        #     return zero_mat.reshape(int(K1 * K2), order="C")
        #
        # def _get_col(col):
        #     #   获取代表某列为1，其他为0的指示向量
        #     zero_mat = np.zeros((K1, K2))
        #     zero_mat[:, col] = 1
        #     return zero_mat.reshape(int(K1 * K2), order="C")
        #
        # P1 = np.array([item[1] for item in spectrum1])
        # P2 = np.array([item[1] for item in spectrum2])
        #
        # P1 = P1 / np.sum(P1)  # 归一化
        # P2 = P2 / np.sum(P2)
        #
        # #   如果spectrum1只有一个簇群，即A是一个行向量，那么直接按照P2分配即可
        # if K1 == 1:
        #     # A = np.expand_dims(P2, axis=0)
        #     # return np.sum(C * A)
        #     return np.sum(P2 * np.reshape(C, -1))
        # #   如果spectrum2只有一个簇群，即A是一个列向量，那么直接按照P1分配即可
        # if K2 == 1:
        #     # A = np.expand_dims(P1, axis=0)
        #     # return np.sum(A*C)
        #     return np.sum(P1 * np.reshape(C, -1))
        #
        # C_vector = C.reshape(int(K1 * K2), order="C")
        #
        # #   A[i,j]表示把谱1的第i簇指派给谱2的第j簇，输入约束是A的列和等于谱B的比例
        # input_constrain = np.array([_get_col(col) for col in range(K2)])
        # output_constrain = np.array([_get_row(row) for row in range(K1)])
        # A_eq = np.concatenate([input_constrain, output_constrain], axis=0)
        # b_eq = np.concatenate([P2, P1])
        #
        # from scipy import optimize as op
        # res = op.linprog(c=C_vector,
        #                  A_eq=A_eq,
        #                  b_eq=b_eq)
        # return res['fun']


#   将色彩谱保存为Excel文件
def spectrum_in_xls(spectrum_paths, target_xls):
    xcs = Xls_Color_Spectrum(target_xls)
    for i, pkl_path in enumerate(spectrum_paths):
        [spectrum] = load_variables(pkl_path)
        xcs.write(row=i, col=0, color=None, content=purename(pkl_path))
        cursor = 1
        cluster_amount = len(spectrum['weights'])
        for j in range(cluster_amount):  # 这里的item表示的就是不同的簇群啦！
            r = spectrum['means'][j, 0]
            g = spectrum['means'][j, 1]
            b = spectrum['means'][j, 2]
            p = spectrum['weights'][j] * 100
            xcs.write(row=i, col=(cursor, cursor + int(p)), color=(r, g, b))
            cursor += int(p)
    xcs.finish()


# #   将聚类结果保存为Excel，cluster_file保存的是具体的数据，spectrum_file保存的是
# def save_clustering_to_xls(pkl_paths, spectrum_data_xls_file_path=None, spectrum_color_xls_file_path=None):
#     if spectrum_data_xls_file_path is not None:
#         from ..utils import WorkBookwriter, load_variables
#         from ..files import purename
#         wbw = WorkBookwriter(target_path=spectrum_data_xls_file_path, worksheet_name='clustering')
#         for pkl_path in pkl_paths:
#             [this_slide_result] = load_variables(pkl_path)
#             wbw.write(purename(pkl_path))
#             for item in this_slide_result:
#                 wbw.write(str(item))
#             wbw.go_to_next_row()
#         wbw.finish()
#     if spectrum_color_xls_file_path is not None:
#         from ..utils import load_variables
#         from ..plotter import Xls_Color_Spectrum
#         xcs = Xls_Color_Spectrum(spectrum_color_xls_file_path)
#         for i, pkl_path in enumerate(pkl_paths):
#             [this_slide_result] = load_variables(pkl_path)
#             cursor = 0
#             for item in this_slide_result:  # 这里的item表示的就是不同的簇群啦！
#                 ((r, g, b), p) = item
#                 xcs.write(row=i, col=(cursor, cursor + int(p)), color=(r, g, b))
#                 cursor += int(p)
#         xcs.finish()


#   在右侧生成bar
#   对于一张图像，显示其颜色分布
#   输入：
#       spectrum：分析好的结果
#       bar_height：显示的颜色条高度（不同颜色构成，在高度上进行切分）
#       bar_width：显示的颜色条宽度
def get_color_bar(spectrum,
                  bar_height=1000, bar_width=100):
    bar_image = np.zeros((bar_height, bar_width, 3), dtype='uint8')
    current_cursor = 0
    weights = spectrum['weights']
    means = spectrum['means']
    n = len(weights)
    for i in range(n):
        this_tile_height = int(bar_height * weights[i])
        bar_image[current_cursor:current_cursor + this_tile_height, :, 0] = np.floor(means[i, 0])
        bar_image[current_cursor:current_cursor + this_tile_height, :, 1] = np.floor(means[i, 1])
        bar_image[current_cursor:current_cursor + this_tile_height, :, 2] = np.floor(means[i, 2])
        current_cursor += this_tile_height
    return bar_image
