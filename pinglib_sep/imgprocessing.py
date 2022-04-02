#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
读取与保存：
    读取：       img = imread(img_path)
    保存：       imwrite(img, img_path)
二值图操作：
    联通域标签：  labeled_array, num_features = bwlabel(bw)
        *   在scipy中默认是按照4邻域的，在matlab中默认是按照8邻域的，这里以8邻域为准
    距离图：     dist_map = bwdist(bw)
    闭运算：     bw = bwclose(bw, iterations=1)
    开运算：     bw = bwopen(bw, iterations=1)
    腐蚀：       bw = bwerode(bw, iterations=1)
    膨胀：       bw = bwdilate(bw, iterations=1)
色彩转换：
    灰度图：     gray_img = rgb2gray(rgb_img)
    转HSV：     hsv_img = rgb2hsv(rgb_img)
滤波：
    高斯滤波器： filter = gaussian(sigma=1.5, size=None)
    卷积：      img = conv2d(img, filter, mode='same')
阈值：
    OTSU：      th = otsu(gray_img)
变换：
    缩放：      img = imrescale(img, scale)
    缩放：      img = imrescale_maintain(img, scale) # 缩放时仍然保持原矩阵大小，多了裁剪，少了填0
    旋转：      img = imrotate(img, angle)     (angle in degrees)
    裁剪：      img = imcrop(img, size, location=None)
增强：          imenhance(img, method, factor) #   可选method（字符串）包括contrast，sharpness，color，brightness

'''

import os
import numpy as np
import scipy.signal
from PIL import Image
from .files import create_dir

'''【增强】'''


#   图像增强，可选用的方法包括：
#       contrast(默认)：增强对比度，0对应灰图，1对应原图，>1表示增强
#       brightness：增强亮度，0对应黑图，1对应原图，>1表示调亮
#       color：色彩均衡，0对应黑白图，1对应原图，>1表示增强
#       sharpness：锐化，0表示模糊图，1表示原图，2表示锐化图
def imenhance(img, method='contrast', factor=1.0):
    from PIL import ImageEnhance
    image = Image.fromarray(img)
    if method == 'sharpness':
        enh = ImageEnhance.Sharpness(image)
    elif method == 'color':
        enh = ImageEnhance.Color(image)
    elif method == 'brightness':
        enh = ImageEnhance.Brightness(image)
    else:
        enh = ImageEnhance.Contrast(image)
    return np.array(enh.enhance(factor))


'''【读取和保存】'''


#   缩放数值，在保存图像时很有用
def value_rescale(img, rescale=None):
    #   如果是None，那么直接返回原数据
    if rescale is None:
        return img
    #   如果是False，也直接返回
    if isinstance(rescale, bool):
        if not rescale:
            return img
        else:
            #   这时候rescale是True，那么自动推断缩放范围
            datamat = np.double(img)
            minval = np.min(datamat)
            maxval = np.max(datamat)
            #   如果只有一个值，那么按照其是否为0，缩放为0或者255
            if minval == maxval:
                if minval > 0:
                    datamat = datamat / maxval * 255
            else:
                #   在其他的情况下就可以放心缩放了
                datamat = (datamat - minval) * 255 / (maxval - minval)
    else:
        #   这时候，rescale应该是一个具体的数值了
        datamat = np.double(img)
        datamat = datamat * rescale
    img = np.array(datamat).astype(np.uint8)
    return img


#   打开一个图像文件，并读入数据
def imread(img_path):
    Image.MAX_IMAGE_PIXELS = None
    return np.array(Image.open(img_path))


#   将数据矩阵存储为图像
#   参数：
#       fill_val：对于img数据中nan的部分，用何值填充
#       rescale：是否对像素值进行缩放（例如label map会希望缩放，否则太暗看不清）
#       rescale_factor：像素值缩放的尺度，如果为None则自动计算，使最大值恰好为255
def imwrite(img, img_path, fill_val=None, rescale=False, rescale_factor=None, **kwargs):
    dir_path, _ = os.path.split(img_path)
    if not (dir_path == '') and not (os.path.isdir(dir_path)):
        create_dir(dir_path)

    if img.dtype == np.bool:
        datamat = (np.array(img).astype(np.double) * 255).astype(np.uint8)
    else:
        datamat = np.array(img).astype(np.uint8)
    if fill_val is not None:
        datamat = fill_nan(datamat, fill_value=fill_val)
    #   如果需要，进行缩放
    if rescale:
        datamat = value_rescale(datamat, rescale_factor)
    #   还要处理一下灰度图的情况（可能第三个维度为1）
    try:
        datamat = np.squeeze(datamat)
    except:
        print('error happens while writing {}'.format(img_path))
    im = Image.fromarray(datamat)
    im.save(img_path, **kwargs)


'''【变换】'''

from scipy.ndimage.interpolation import zoom as scipy_ndi_int_zoom


#   搞这么麻烦，是因为不想在对三维图缩放的时候，把第三维也缩放了……
def imrescale(img, scale, hard=False):
    target_width = int(np.round(img.shape[1] * scale))
    target_height = int(np.round(img.shape[0] * scale))
    return imrescale_to_shape(img, (target_height, target_width), hard=hard)


#   保持图像原有尺寸的情况下进行缩放，如果缩小的话，将填充default_val
def imrescale_maintain(img, scale, default_val=0, hard=False):
    img_height = img.shape[0]
    img_width = img.shape[1]
    if scale > 1:
        bigger_img = imrescale(img, scale, hard)
        return imcrop(bigger_img, (img_width, img_height))
    else:
        canvas = np.ones_like(img) * default_val
        smaller_img = imrescale(img, scale, hard)
        smaller_img_height = smaller_img.shape[0]
        smaller_img_width = smaller_img.shape[1]
        left_x = int(np.floor(img_width / 2 - smaller_img_width / 2))
        top_y = int(np.floor(img_height / 2 - smaller_img_height / 2))
        canvas[top_y:top_y + smaller_img_height, left_x:left_x + smaller_img_width] = smaller_img
        return canvas
    #   其实并不需要对二值图进行特殊的讨论，这个应该在调用时判断


#   把图像缩放到指定尺寸(使用线性插值，会产生过度) 这里的尺寸是矩阵的尺寸，也即高度（行）在前，宽度（列）在后
def imrescale_to_shape(img, target_shape, hard=False):
    # img_shape = img.shape
    # final_scale = np.ones_like(img_shape).astype(np.double)
    # final_scale[0] = target_shape[0] / img_shape[0]
    # final_scale[1] = target_shape[1] / img_shape[1]
    # return scipy_ndi_int_zoom(img, final_scale, order=1, mode='nearest')
    if isinstance(target_shape, int):
        target_shape = (target_shape, target_shape)
    #   如果图像的尺寸与要求的相符，直接返回图像即可（若仍按照流程缩放亦可行，但尺寸相同情况下的resize似乎会浪费大量的时间）
    original_shape = img.shape
    if original_shape[0] == target_shape[0] and original_shape[1] == target_shape[1]:
        return img

    try:
        img = Image.fromarray(img)
        if hard:
            return np.array(img.resize((target_shape[1], target_shape[0]), Image.NEAREST))
        else:
            return np.array(img.resize((target_shape[1], target_shape[0]), Image.BILINEAR))
    except:
        #   此时，说明img不能被作为图像
        from scipy.ndimage.interpolation import zoom
        return zoom(img, [target_shape[0] / original_shape[0], target_shape[1] / original_shape[1], 1], order=1,
                    mode='nearest')


#   把图像缩放到指定尺寸(使用硬过渡，不会产生中间值)，此功能已集成到imrescale里面了
# def imrescale_to_shape_hard(img, target_shape):
#     img = Image.fromarray(img)
#     return np.array(img.resize((target_shape[1], target_shape[0]), Image.NEAREST))


#   将图像扩大n倍（不插值）
def imrepeat(img, repeats):
    return np.repeat(np.repeat(img, repeats=repeats, axis=0), repeats=repeats, axis=1)


# def imrescale(img, scale, is_bw_img=False):
#     from scipy.ndimage.interpolation import zoom as scipy_ndi_int_zoom
#     if np.max(np.double(img)) <= 1:
#         #   这时候是二值图，需要特殊处理
#         img = np.uint8(np.double(img) * 255)
#         scaled_img = scipy_ndi_int_zoom(img, scale) > 0
#     else:
#         scaled_img = scipy_ndi_int_zoom(img, scale)
#     if is_bw_img:
#         scaled_img = scaled_img > 0
#     return scaled_img


def imrotate(img, angle):
    from scipy.ndimage.interpolation import rotate as scipy_ndi_int_rotate  # 防止命名冲突
    return scipy_ndi_int_rotate(img, angle)


#   注意，如果你输入的是0~1的小数，那么在最后一句话上转换类型时会损失精度！（已修正）
def imcrop(img, size, position=None):
    [img_h, img_w] = img.shape[0:2]
    if position is None:
        position = (np.floor(img_w / 2), np.floor(img_h / 2))
    if isinstance(size, int):
        w = size
        h = size
    else:
        w = size[0]
        h = size[1]
    x = int(position[0] - np.floor(w / 2))
    y = int(position[1] - np.floor(h / 2))
    # im = Image.fromarray(np.array(img).astype(np.uint8)).crop((x, y, x + w, y + h))
    # return np.array(im)   #   这两句话会在0~1的小数矩阵上损失精度！
    return img[y:y + h, x:x + w]


#   对二维图使用pad或crop使其成为指定的尺寸
def pad_or_crop(img, target_size):
    if len(img.shape) == 3:
        img = img[..., 0]
    pad_left, pad_right, pad_top, pad_bottom = 0, 0, 0, 0
    if img.shape[0] < target_size[0]:
        delta = target_size[0] - img.shape[0]
        pad_top = delta // 2
        pad_bottom = delta - pad_top
    if img.shape[1] < target_size[1]:
        delta = target_size[0] - img.shape[0]
        pad_left = delta // 2
        pad_right = delta - pad_left
    img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant')
    img = imcrop(img, target_size)
    return img


'''【二值图操作】'''


#   打标签：labeled_array, num_features = bwlabel(bwimg)
def bwlabel(bwimg):
    from scipy.ndimage.measurements import label as scipy_ndi_mea_label
    return scipy_ndi_mea_label(bwimg, np.ones((3, 3)))


#   距离变换
from scipy.ndimage.morphology import distance_transform_edt as bwdist

#   形态学运算
from scipy.ndimage.morphology import binary_closing, binary_dilation, binary_erosion, binary_opening


#   闭运算


def bwclose(input, iterations=1):
    return binary_closing(input, iterations=iterations)


#   开运算
def bwopen(input, iterations=1):
    return binary_opening(input, iterations=iterations)


#   腐蚀
def bwerode(input, iterations=1):
    return binary_erosion(input, iterations=iterations)


#   膨胀
def bwdilate(input, iterations=1):
    return binary_dilation(input, iterations=iterations)


'''【色彩转换】'''


#   将一幅rgb图像转成灰度图
def rgb2gray(img):
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


#   将一幅图像从rgb空间(0~255)转换到hsv空间(0~255)
def rgb2hsv(rgb_img):
    import colorsys
    img_width, img_height, _ = rgb_img.shape
    rgb_img = rgb_img / 255.0  # colorsys接受的rgb在0~1
    hsv_img = np.zeros([img_width, img_height, 3])
    for i in range(img_width):
        for j in range(img_height):
            hsv_img[i, j, :] = np.array(colorsys.rgb_to_hsv(rgb_img[i, j, 0], rgb_img[i, j, 1], rgb_img[i, j, 2]))
    hsv_img = hsv_img * 255
    return hsv_img


'''【滤波】'''


def get_gaussian_filter(sigma=1.5, size=None):
    #   如果只给了sigma，就按 3 sigma准则配置尺寸
    if size is None:
        size = np.ceil(3 * sigma + 1).astype(np.integer)
    x, y = np.meshgrid(range(-size, size + 1), range(-size, size + 1))
    d2 = np.square(x) + np.square(y)
    filter = np.exp(-d2 / (2 * sigma * sigma)) / (2 * np.pi * sigma)
    return filter / np.sum(filter)


def conv2d(img, filter, mode='same'):
    if len(img.shape) == 3:
        r = scipy.signal.fftconvolve(img[:, :, 0], filter, mode=mode)
        g = scipy.signal.fftconvolve(img[:, :, 1], filter, mode=mode)
        b = scipy.signal.fftconvolve(img[:, :, 2], filter, mode=mode)
        r = r[:, :, np.newaxis]
        g = g[:, :, np.newaxis]
        b = b[:, :, np.newaxis]
        return np.concatenate((r, g, b), axis=2)
    else:
        return scipy.signal.fftconvolve(img, filter, mode=mode)


from scipy.ndimage import gaussian_filter

'''【阈值】'''


#   这里的img应该是灰度图
def otsu(img):
    from skimage import filters
    if len(img.shape) == 3:
        img_gray = rgb2gray(img)
    else:
        img_gray = img
    return filters.threshold_otsu(img_gray)


'''【填充】'''


def fill_nan(img, fill_value=0):
    img[np.isnan(img)] = fill_value
    return img


'''【维度】'''


#   把一个图像从3维变为2维（取首个维度），如果本来就是二维，直接返回
def imsqueeze(img):
    if len(img.shape) == 3:
        return img[:, :, 0]
    else:
        return img


#   把一个图像从2维扩展为3维灰度图，如果本来就是三维，直接返回
def imexpanddim(img):
    if len(img.shape) == 2:
        return np.repeat(np.expand_dims(img, axis=2), repeats=3, axis=2)
    else:
        return img


'''【推荐的颜色】'''


#   定义一些具有显著差别的颜色(稍有不同，尽量贴合ASAP习惯)
#   参考：https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
def get_rgb_table(ASAP_habit=False):
    if ASAP_habit:
        rgb_table = [
            [0, 0, 0],
            [0, 130, 200],
            [170, 255, 195],
            [255, 255, 25],
            [245, 130, 48],
            [250, 190, 190],
            [230, 190, 255],
            [0, 0, 128],
            [170, 110, 40],
            [230, 25, 75],
            [0, 128, 128],
            [60, 128, 75],
            [70, 240, 240],
            [240, 50, 230],
            [255, 250, 200],
            [128, 0, 0],
            [128, 128, 0],
            [210, 245, 60],
            [145, 30, 180],
            [255, 215, 180]
        ]
    else:
        rgb_table = [
            [0, 0, 0],
            [0, 130, 200],
            [128, 0, 0],
            [0, 0, 128],
            [245, 130, 48],
            [250, 190, 190],
            [230, 190, 255],
            [255, 255, 25],
            [170, 110, 40],
            [230, 25, 75],
            [0, 128, 128],
            [60, 128, 75],
            [70, 240, 240],
            [240, 50, 230],
            [255, 250, 200],
            [170, 255, 195],
            [128, 128, 0],
            [210, 245, 60],
            [145, 30, 180],
            [255, 215, 180]
        ]
    #   后面再多，就用随机的颜色了
    rgb_table = np.concatenate([rgb_table, np.floor(256 * np.random.random_sample((256 - len(rgb_table), 3)))])
    return np.array(rgb_table)


#   获取一些颜色，以16进制字符表示
def get_rgb_hex(index):
    # rgb_table = [[255, 0, 0], [0, 255, 0]]
    rgb_table = [
        [0, 130, 200],
        [128, 0, 0],
        [0, 0, 128],
        [245, 130, 48],
        [250, 190, 190],
        [230, 190, 255],
        [255, 255, 25],
        [170, 110, 40],
        [230, 25, 75],
        [0, 128, 128],
        [60, 128, 75],
        [70, 240, 240],
        [240, 50, 230],
        [255, 250, 200],
        [170, 255, 195],
        [128, 128, 0],
        [210, 245, 60],
        [145, 30, 180],
        [255, 215, 180]]
    strs = '#'
    index = index % len(rgb_table)
    for num in rgb_table[index]:
        # 将R、G、B分别转化为16进制拼接转换并大写
        strs += str(hex(num))[-2:].replace('x', '0').upper()
    return strs


def color_schemes(type):
    if isinstance(type, np.ndarray):
        return type
    if type == 'non-tumor/tumor':
        #   靛蓝色、红色
        rgb_table = [[0, 0, 0],
                     [0, 174, 202],
                     [241, 82, 63]]

    elif type == 'breast-4':
        #   'tumor', 'stroma', 'inflammatory', 'necrosis'
        #   红色      黄色       靛蓝色            紫红色
        rgb_table = [[0, 0, 0],
                     [241, 82, 63],
                     [254, 218, 64],
                     [0, 174, 202],
                     [187, 128, 184],
                     [0, 0, 0]]
    elif type == 'breast-7':
        # #   'benign', 'tumor', 'necrosis', 'stroma', 'inflammatory', 'ADI', 'BAC'
        # #   橙色       红色     紫红色       黄色        靛蓝色         灰色    白色
        # rgb_table = [[0, 0, 0],
        #              [241, 153, 93],
        #              [241, 82, 63],
        #              [187, 128, 184],
        #              [254, 218, 64],
        #              [0, 174, 202],
        #              [191, 191, 191],
        #              [255, 255, 255]]
        #   'tumor', 'debris', 'stroma', 'inflammatory', 'adipose', 'mucus', 'background'
        #   红色      紫红色     黄色      靛蓝色           灰色       嫩绿色    白色
        rgb_table = [[0, 0, 0],
                     [241, 82, 63],
                     [187, 128, 184],
                     [254, 218, 64],
                     [0, 174, 202],
                     [191, 191, 191],
                     [121, 231, 11],
                     [255, 255, 255]]

    elif type == 'crc-8':
        #   ADI   DEB   LYM   MUC   MUS   NRM   STR   TUM
        #   灰色  紫红色 靛蓝色 淡青色 橙色  嫩绿色 黄色  红色
        rgb_table = [[0, 0, 0],
                     [191, 191, 191],
                     [187, 128, 184],
                     [0, 174, 202],
                     [131, 207, 198],
                     [241, 153, 93],
                     [121, 231, 11],
                     [254, 218, 64],
                     [241, 82, 63]]
    elif type == 'crc-9':
        #   ADI   BAC   DEB   LYM   MUC   MUS   NRM   STR   TUM
        #   灰色  白色  紫红色 靛蓝色 淡青色 橙色  嫩绿色 黄色  红色
        rgb_table = [[0, 0, 0],
                     [191, 191, 191],
                     [255, 255, 255],
                     [187, 128, 184],
                     [0, 174, 202],
                     [131, 207, 198],
                     [241, 153, 93],
                     [121, 231, 11],
                     [254, 218, 64],
                     [241, 82, 63]]
    elif type == 'glioma-8':
        #
        rgb_table = [[255, 255, 255],
                     [255, 0, 0],
                     [204, 0, 153],
                     [255, 102, 0],
                     [0, 153, 255],
                     [0, 227, 222],
                     [51, 204, 51],
                     [0, 128, 0],
                     [255, 255, 0]]
    elif type == 'glioma-9':
        rgb_table = [[255, 255, 255],
                     [255, 0, 0],
                     [255, 80, 80],
                     [191, 191, 191],
                     [0, 153, 255],
                     [0, 153, 255],
                     [255, 217, 102],
                     [255, 255, 0],
                     [209, 4, 208]]
    else:
        raise NotImplementedError
    pass
    return np.array(rgb_table)


'''【越界采样】'''


def patch_from_image(image, x, y, scale, size, padding=(0, 0, 0), mode='center', with_valid_flag=False):
    image_height = image.shape[0]
    image_width = image.shape[1]
    dim = len(image.shape)

    #   关键是，我们要知道从原图的哪里截，截多少，填充到哪里
    if not isinstance(size, list) and not isinstance(size, tuple):
        scaled_size = (int(size / scale), int(size / scale))  # 如果不是list的形式，则是单个的数
    else:
        scaled_size = (int(size[0] / scale), int(size[1] / scale))  # 比如scale=0.5，那么在原图上截的范围就会是2倍，所以这里是除
    if mode == 'center':
        x_lb = x - scaled_size[0] // 2
        y_lb = y - scaled_size[1] // 2
    else:
        x_lb = x
        y_lb = y
    x_ub = x_lb + scaled_size[0]
    y_ub = y_lb + scaled_size[1]  # 计算出了在原图中截取的范围，但是，它们有可能是越界的，怎样保证它返回一个scaled_size的patch呢？
    valid_flag = x_lb >= 0 and y_lb >= 0 and x_ub < image_width and y_ub < image_height

    #   这样吧，我们构建一个大一点的canvas，然后只需要计算出适当的位置，往上面写信息就行了
    if dim == 3:
        image_patch_canvas = np.zeros((scaled_size[1], scaled_size[0], image.shape[2]), dtype='uint8')
        if (not isinstance(padding, tuple)) and (not isinstance(padding, list)):
            padding = (padding, padding, padding)
        for i in range(image.shape[2]):
            image_patch_canvas[:, :, i] = padding[i]
    else:
        if isinstance(padding, tuple) or isinstance(padding, list):
            image_patch_canvas = np.ones((scaled_size[1], scaled_size[0]), dtype='uint8') * padding[0]
        else:
            image_patch_canvas = np.ones((scaled_size[1], scaled_size[0]), dtype='uint8') * padding

    #   这是在原图上采样实际生效的位置
    x_lb_effect = max(0, x_lb)
    y_lb_effect = max(0, y_lb)
    x_ub_effect = min(image_width, x_ub)
    y_ub_effect = min(image_height, y_ub)

    #   如果x_lb没有越界的行为，写入的起始位置就是0，但是如果小于0，写入位置就会右移
    #   这里是在canvas上写入的位置
    x_lb_write = 0 if x_lb >= 0 else -x_lb
    y_lb_write = 0 if y_lb >= 0 else -y_lb
    if x_lb_write >= image_patch_canvas.shape[1] or y_lb_write >= image_patch_canvas.shape[0]:
        #   这里是做一个过滤，以免越界写大小为0的块，会报错
        pass
    else:
        if dim == 3:
            image_patch_canvas[y_lb_write:y_lb_write + y_ub_effect - y_lb_effect,
            x_lb_write:x_lb_write + x_ub_effect - x_lb_effect, :] = \
                image[y_lb_effect:y_ub_effect, x_lb_effect:x_ub_effect, :]
        else:
            image_patch_canvas[y_lb_write:y_lb_write + y_ub_effect - y_lb_effect,
            x_lb_write:x_lb_write + x_ub_effect - x_lb_effect] = \
                image[y_lb_effect:y_ub_effect, x_lb_effect:x_ub_effect]

    #   然后我们需要把这个canvas缩放到原先的尺寸
    if isinstance(size, tuple) or isinstance(size, list):
        result_image_patch = imrescale_to_shape(image_patch_canvas, (size[0], size[1]), hard=True)
    else:
        result_image_patch = imrescale_to_shape(image_patch_canvas, (size, size), hard=True)
    if with_valid_flag:
        return result_image_patch, valid_flag
    else:
        return result_image_patch


'''【ImageNet预处理】'''


#   输入一个3xhxw的tensor，返回一个uint8矩阵，主要是方便调试用的
def imagenet_preprocess_tensor_to_uint8(image_tensor):
    image_uint8 = image_tensor.transpose(1, 2, 0)
    mean_val = [0.485, 0.456, 0.406]
    std_val = [0.229, 0.224, 0.225]
    for i in range(3):
        image_uint8[:, :, i] = (image_uint8[:, :, i] * std_val[i]) + mean_val[i]
    image_uint8[image_uint8 > 1] = 1
    image_uint8[image_uint8 < 0] = 0
    image_uint8 = (image_uint8 * 255).astype(np.uint8)
    return image_uint8


#   输入一个hxwx3的uint8图像矩阵，返回一个归一化的3xhxw的矩阵
def imagenet_preprocess_uint8_to_tensor(image_uint8):
    image_tensor = image_uint8 / 255
    mean_val = [0.485, 0.456, 0.406]
    std_val = [0.229, 0.224, 0.225]
    for i in range(3):
        image_tensor[:, :, i] = (image_tensor[:, :, i] - mean_val[i]) / std_val[i]
    image_tensor = image_tensor.transpose(2, 0, 1)
    return image_tensor


#   直接使用RGB色彩空间进行过滤
#   如果使用大于零的数，则超过该值才认为是前景；如果使用小于零的数，如-20，则小于20才认为是前景
#   如果使用的是小数，则表示按百分位数过滤，大于零的小数作为下限；小于零的小数作为上限。如-0.3表示强度在30%之下才作为前景
#   当使用多个判断条件时，各成分以“与”的方式组合在一起
def rgb_threshold(img, r_th=None, g_th=None, b_th=None, aggregate='and'):
    def th_on_one_channel(channel_img, val):
        if (val is None) or val == 0:
            return np.ones_like(channel_img).astype(np.bool)
        if val >= 1:
            return channel_img > val
        if val <= -1:
            return channel_img < -val
        if val > 0:
            return channel_img > np.percentile(channel_img, val)
        if val < 0:
            return channel_img < np.percentile(channel_img, -val)

    if aggregate == 'and':
        mask = th_on_one_channel(img[:, :, 0], r_th)
        mask = np.logical_and(mask, th_on_one_channel(img[:, :, 1], g_th))
        mask = np.logical_and(mask, th_on_one_channel(img[:, :, 2], b_th))
    else:
        mask = th_on_one_channel(img[:, :, 0], r_th)
        mask = np.logical_or(mask, th_on_one_channel(img[:, :, 1], g_th))
        mask = np.logical_or(mask, th_on_one_channel(img[:, :, 2], b_th))
    return mask
