#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import xlsxwriter
from .imgprocessing import imexpanddim, imrescale_to_shape
from copy import deepcopy


class Xls_Color_Spectrum():
    def __init__(self, target_file='color_spectrum.xlsx', col_amount=100, col_width=0.3):
        self.workbook = xlsxwriter.Workbook(target_file)
        self.worksheet = self.workbook.add_worksheet()
        self.worksheet.set_column(first_col=1, last_col=col_amount, width=col_width)

    def __rgb2hex(self, rgb):
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        return '#%02x%02x%02x' % (r, g, b)

    def write(self, row, col, color=None, content=None):
        #   这里会反复地增加format，不知道会不会影响容量，系统会不会识别重复项，如果有问题的话，自己设计一个过滤器
        content = None if content is None else str(content)
        format = self.workbook.add_format(
            {'pattern': 1, 'bg_color': self.__rgb2hex(color)}) if color is not None else None
        if not isinstance(row, tuple):
            row = (row, row + 1)
        if not isinstance(col, tuple):
            col = (col, col + 1)
        for r in range(*row):
            for j in range(*col):
                if format is not None:
                    self.worksheet.write(r, j, content, format)
                else:
                    self.worksheet.write(r, j, content)

    def finish(self):
        self.workbook.close()


#   将rgb图像绘制在指定的坐标，通常用于特征可视化
#   输入参数：
#       locations：Nx2的numpy数组，表示N张图片的位置
#       image_sources：由N张图片路径组成的list
#       size：每张image占整幅图像的长度比例
#       highlight_flags：N长度的numpy数组，若非0，则对应图像将以红边框显示
#       savepath：图像保存路径，若为None，则直接显示
#       shuffle：是否对图像进行乱序
def figures_as_scatter(locations, image_sources, highlight_flags=None, size=0.05, figsize=8, savepath=None,
                       shuffle=False, ratio=1.0):
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np

    #   首先将location折算到0~1内
    minval = np.min(locations, axis=0)
    maxval = np.max(locations, axis=0)
    x = (locations[:, 0] - minval[0]) / (maxval[0] - minval[0])
    y = (locations[:, 1] - minval[1]) / (maxval[1] - minval[1])

    #   让小图的另一端不会越过边界
    x = x * (1 - size)
    y = y * (1 - size)

    #   创建画布
    figure = plt.figure(1, figsize=(figsize, figsize))

    #
    image_amount = len(image_sources)
    shuffle_index = np.arange(image_amount)
    if shuffle:
        np.random.shuffle(shuffle_index)

    #   依次绘制小图片
    for i in range(image_amount):

        if np.random.random_sample() >= ratio:
            continue

        index = shuffle_index[i]

        #   创建并配置Axes
        axes = figure.add_axes([x[index], y[index], size, size], frameon=True)
        axes.grid(False)
        axes.set_xticks([])
        axes.set_yticks([])
        axes.axis('equal')

        #   配置Axes的边框
        if highlight_flags is not None and highlight_flags[index] > 0:
            axes.spines['bottom'].set_color('red')
            axes.spines['top'].set_color('red')
            axes.spines['right'].set_color('red')
            axes.spines['left'].set_color('red')

        #   显示图像
        image_data = np.array(Image.open(image_sources[index]))
        axes.imshow(image_data)

    #   按照路径配置，显示或者直接保存图像
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath)


#   把轮廓图显示在rgb图片上，返回绘制过轮廓的RGB图像数据
def get_contoured_image(image_patch, binary_mask, line_width=10):
    from imgprocessing import bwerode
    mask_bw_erode = bwerode(binary_mask, iterations=line_width)
    contour_bw = (binary_mask.astype(np.int) - mask_bw_erode.astype(np.int)).astype(np.bool)
    image_patch[contour_bw] = [255, 0, 0]
    return image_patch


#   把mask转化为彩色的图像~
#   当需要观察特定数值的区域时，将target_value设置为对应值，符合的区域将以白色显示
def _mask_to_color(datamat, target_value=None, color_table=None):
    import numpy as np
    from pinglib.imgprocessing.basic import imsqueeze, get_rgb_table, color_schemes
    if len(datamat.shape) == 3:
        mask = imsqueeze(datamat)
    else:
        mask = datamat
    if color_table is None:
        color_table = get_rgb_table()
    elif isinstance(color_table, str):
        color_table = color_schemes(color_table)
    mask_color = color_table[mask.astype(np.int) % 256]
    if target_value is not None:
        for i in range(3):
            original_channel = datamat[:, :, i]
            colored_channel = mask_color[:, :, i]
            colored_channel[original_channel == target_value] = 255
            mask_color[:, :, i] = colored_channel
    return mask_color.astype(np.uint8)


# 以图像的方式对数据矩阵进行可视化
#   如果设置rescale为True，则将自动归一化到0~255，否则将以原始数据展示
def see(datamat, color_mapping=False, title=None, new_figure=True, show=True,
        value_name_dict=None, **kwargs):
    import numpy as np
    import matplotlib.pyplot as plt
    item = None
    if not isinstance(datamat, np.ndarray):
        datamat = np.array(datamat)
    try:
        if color_mapping:
            datamat = _mask_to_color(datamat)
        if new_figure:
            plt.figure(**kwargs)
        if len(datamat.shape) == 2:
            item = plt.imshow(np.squeeze(datamat), plt.cm.gray)
        else:
            item = plt.imshow(np.squeeze(datamat))
        if title is not None:
            plt.title(title)
        if color_mapping and value_name_dict:
            #   这时候再建一张图用来写颜色
            legend_height = 800
            legend_width = 100
            legend_mat = np.zeros((legend_height, legend_width))
            class_num = len(value_name_dict.keys())
            for i, value in enumerate(value_name_dict.keys()):
                legend_mat[i * legend_height // class_num:(i + 1) * legend_height // class_num, :] = value
            legend_mat = _mask_to_color(legend_mat)
            plt.figure()
            item = plt.imshow(legend_mat)
            for i, value in enumerate(value_name_dict.keys()):
                plt.text(0, legend_height * (i + 0.5) / class_num, value_name_dict[value])
        if show:
            plt.show()
    except Exception as e:
        print('* Error happens during visualization: {}'.format(e))
    return item


class Image_Gird():
    def __init__(self, patch_width=None, patch_height=None,
                 horizontal_amount=1, vertical_amount=1,
                 margin_size=0, margin_color='w',
                 auto_rescale=False):

        self.horizontal_amount = horizontal_amount
        self.vertical_amount = vertical_amount
        self.auto_rescale = auto_rescale
        self.margin_size = margin_size
        self.initialized = False
        if patch_width is not None and patch_height is not None:
            self.patch_width = patch_width
            self.patch_height = patch_height
            if margin_color == 'w' or margin_color == 'W':
                default_val = 255
            else:
                default_val = 0
            self.data = default_val * np.ones(
                (int(patch_height * self.vertical_amount) + (self.vertical_amount - 1) * self.margin_size,
                 int(patch_width * self.horizontal_amount) + (self.horizontal_amount - 1) * self.margin_size, 3),
                dtype='uint8')
            self.initialized = True
        else:
            #   如果没指定大小，就没法auto rescale了
            assert not auto_rescale

    def _data_initialize(self, patch_width, patch_height):
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.data = 255 * np.ones(
            (int(patch_height * self.vertical_amount) + (self.vertical_amount - 1) * self.margin_size,
             int(patch_width * self.horizontal_amount) + (self.horizontal_amount - 1) * self.margin_size, 3),
            dtype='uint8')
        self.initialized = True
        pass

    #   对一个格子写图像数据，location可以是(0,0)，表示第0行第1列，也可以是一个数，比如10表示第1行第0列
    def write(self, input_image_data, location=(0, 0), auto_rescale=True):
        image_data = deepcopy(input_image_data)
        if len(input_image_data.shape) == 2:
            image_data = imexpanddim(image_data)
        #   检查是否已初始化
        if not self.initialized:
            patch_height, patch_width = image_data.shape[0:2]
            self._data_initialize(patch_width, patch_height)
        #   如果没有为此次写入声明缩放方式，则使用对象默认的定义
        if auto_rescale is None:
            auto_rescale = self.auto_rescale
        #   检查是否符合尺寸要求
        if not (image_data.shape[0] == self.patch_height and image_data.shape[1] == self.patch_width):
            if auto_rescale:
                #   如果尺寸不匹配，且启用了缩放，则进行缩放（以及padding） TODO
                image_height = image_data.shape[0]
                image_width = image_data.shape[1]
                height_ratio = image_height / self.patch_height
                width_ratio = image_width / self.patch_width
                effect_ratio = max(height_ratio, width_ratio)
                after_width = int(image_width / effect_ratio)
                after_height = int(image_height / effect_ratio)
                if after_width > self.patch_width:
                    after_width = self.patch_width
                if after_height > self.patch_height:
                    after_height = self.patch_height
                image_data = imrescale_to_shape(image_data, (after_height, after_width), hard=True)
                canvas = np.zeros((self.patch_height, self.patch_width, 3), dtype='uint8')
                canvas[:after_height, :after_width, :] = image_data
                image_data = canvas
            else:
                print('Shape not match expect for (h, w) = ({}, {}), but got ({}, {})'.
                      format(self.patch_height, self.patch_width, image_data.shape[0], image_data.shape[1]))
                raise ValueError
        #   搞清楚写的位置，如果送进来的是一个数，比如11，应当知道第0行是0~9，第1行是10,11,...，所以位于第1行，第1列
        if not isinstance(location, tuple):
            row_id = location // self.horizontal_amount
            col_id = location - row_id * self.horizontal_amount
        else:
            row_id = location[0]
            col_id = location[1]
        #   快乐地写数据
        self.data[
        row_id * self.patch_height + self.margin_size * row_id:
        (row_id + 1) * self.patch_height + self.margin_size * row_id,
        col_id * self.patch_width + self.margin_size * col_id:
        (col_id + 1) * self.patch_width + self.margin_size * col_id, :] = image_data

    #   读取某一个格子的数据
    def load(self, location):
        if not isinstance(location, tuple):
            row_id = location // self.horizontal_amount
            col_id = location - row_id * self.horizontal_amount
        else:
            row_id = location[0]
            col_id = location[1]
        return self.data[
               row_id * self.patch_height + self.margin_size * row_id:
               (row_id + 1) * self.patch_height + self.margin_size * row_id,
               col_id * self.patch_width + self.margin_size * col_id
               :(col_id + 1) * self.patch_width + self.margin_size * col_id, :]

    #   读取所有的数据，打包在一起
    def load_all(self):
        data_list = []
        for r_id in range(self.vertical_amount):
            for c_id in range(self.horizontal_amount):
                data_list.append(self.load((r_id, c_id)))
        return np.concatenate(np.expand_dims(data_list, axis=0))

    #   将自己写到一个pkl文件中
    def save_state(self, path):
        from .utils import save_variables
        save_variables(
            [self.patch_width, self.patch_height, self.horizontal_amount, self.vertical_amount, self.auto_rescale,
             self.data], path)

    #   从pkl中载入状态
    def load_state(self, path):
        from .utils import load_variables
        [self.patch_width, self.patch_height, self.horizontal_amount, self.vertical_amount, self.auto_rescale,
         self.data] = load_variables(path)

    def show(self):
        see(self.data)

    def save(self, path, **kwards):
        #   注：这一部分内容其实和imgprocessing.basic.imwrite重复了，无所谓啦
        from matplotlib import pyplot as plt
        plt.figure()
        see(self.data, new_figure=False, show=False)
        img_height, img_width = self.data.shape[0:2]
        dpi = 300
        plt.axis('off')
        fig = plt.gcf()
        fig.set_size_inches(img_width / dpi, img_height / dpi)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(path, dpi=dpi, pad_inches=0, **kwards)
        #   These code move the white margin and axis, refer to https://blog.csdn.net/jifaley/article/details/79687000
        plt.close()


if __name__ == "__main__":
    pass
