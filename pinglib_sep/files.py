# !/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np


#   获取一个文件夹下所有的文件(仅限根目录，非递归)，排序后形成路径的list
def get_file_list(path, ext='', contain_string=None):
    import os
    if isinstance(path, list):
        result = []
        for item in path:
            result += get_file_list(item, ext)
        return result
    else:
        if contain_string is None:
            result = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)])
        else:
            result = sorted(
                [os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext) and f.find(contain_string) > -1])
        try:
            from tkinter import Tcl
            return list(Tcl().call('lsort', '-dict', result))
        except:
            # print('Error happens at [get_file_list] due to [tkinter]: {}'.format(format_exc()))
            # print('Use default [sorted] instead')
            return result


#   获取一个文件夹下所有的文件（含子目录，递归），排序后形成路径的list
def get_file_list_recursive(path, ext=''):
    file_list = []
    if isinstance(path, list):
        for item in path:
            file_list += get_file_list_recursive(item, ext)
    else:
        for file in os.listdir(path):
            # print(file)
            filepath = os.path.join(path, file)
            # print(filepath)
            if os.path.isdir(filepath):
                file_list = file_list + get_file_list_recursive(filepath, ext)
            else:
                if filepath.endswith(ext):
                    file_list.append(filepath)
    return file_list


#   按照文件名后缀对文件进行匹配
#   输入：
#       filelistList: 多个文件系列组成的list
#       cutoffList,appendixList: 配对的规则是，对应的纯文件名（不含后缀）
#   例如 tumor_01.tiff  --  tumor_01_mask.tiff
#   则appendixList输入['_mask']
def match_files(filelistList, appendixList=None, suffixList=None, verbose=False):
    # def endswith_capital_insensitive(fullstr, endstr):
    #     return fullstr.lower().endswith(endstr.lower())

    #   要求入参是list的形式，如果送入的是字符串，自动先转换为list
    if isinstance(filelistList, str):
        appendixList = [filelistList]
    if isinstance(appendixList, str):
        appendixList = [appendixList]
    if isinstance(suffixList, str):
        suffixList = [suffixList]
    #   如果appendixList留默认为None，则默认根据纯文件名配准
    if appendixList is None:
        appendixList = [''] * len(filelistList)
    #   检查appendixList与filelistList的长度，如果不一致，说明appendix略去了第一个，那么在最前面的位置补''
    if len(filelistList) == len(appendixList) + 1:
        appendixList = [''] + appendixList
    #   现在检查列表长度的数目，如果不一致就报错
    if not (len(filelistList) == len(appendixList)):
        print('Input lists length not agree ({} for filelist, {} for appendix)'.format(len(filelistList),
                                                                                       len(appendixList)))
    if (suffixList is not None) and (len(filelistList) != len(suffixList)):
        print('Input lists length not agree ({} for appendix, {} for suffix)'.format(len(appendixList),
                                                                                     len(suffixList)))
    for i, filelist in enumerate(filelistList):
        if len(filelist) == 0:
            print('Group {} have no component'.format(i))
            raise (ValueError)

    list_length = len(filelistList)
    matchedFilelistList = [[] for _ in range(list_length)]  # 初始化结果列表
    #   把参与运算的变量全部变成小写字符
    from copy import deepcopy
    filelistList_lower = deepcopy(filelistList)
    for i, item_list in enumerate(filelistList):
        for j, item in enumerate(item_list):
            filelistList_lower[i][j] = item.lower()

    for j, item in enumerate(appendixList):
        appendixList[j] = item.lower()
    if suffixList is not None:
        for j, item in enumerate(suffixList):
            suffixList[j] = item.lower()

    # purenamelistList = [[purename(path) for path in filelistList_lower[i]] for i in range(0, list_length)]  # .lower()
    purenamelistList = []
    for i in range(0, list_length):
        # if left_truncated is not None and left_truncated[i] is not None:
        #     current_series = [purename(path)[0:left_truncated[i]] for path in filelistList_lower[i]]
        # else:
        current_series = [purename(path) for path in filelistList_lower[i]]
        purenamelistList.append(current_series)

    #   现在，对每一个文件进行处理
    for k, path in enumerate(filelistList_lower[0]):
        #   如果施加了拓展名限制，则首先验证拓展名
        if suffixList is not None:
            if not path.endswith(suffixList[0]):  # lower()lower()
                continue
        #   获取纯文件名，并把appendix去除掉
        ref_purename = purename(path).replace(appendixList[0], '')  # .lower()
        # ref_purename=purenamelistList[0][k]
        #   开始依次查找
        item_buffer = []  # 用于缓存，如果命中了，这里的内容会依次写到purenamelistList
        valid = True  # 是否全部成功
        #   现在在每个list里面依次查找对应项是否存在
        for i in range(1, list_length):
            search_item = ref_purename + appendixList[i]  # 要查找的项应该是：扣掉尾端的首文件名，加上对应的后缀
            if search_item in purenamelistList[i]:  # .lower()
                #   走到这里的时候，纯文件名已经匹配成功了,现在去filelist中间查找
                index_list = []
                for j, item in enumerate(purenamelistList[i]):
                    if item == search_item:  # .lower().lower()
                        if suffixList is not None:
                            #   如果这时候给定了拓展名，那么还要满足拓展名的约束
                            if not filelistList_lower[i][j].endswith(suffixList[i]):  # lower().lower()
                                continue
                        index_list.append(j)
                if len(index_list) >= 2:
                    print('Error. Can not distinguish {} from {}'.format(filelistList[i][index_list[0]],
                                                                         filelistList[i][index_list[1]]))
                    raise (ValueError)
                item_buffer.append(filelistList[i][index_list[0]])
            else:
                valid = False
                break
        #   现在，如果是合法的结果，那么就加进最终的列表中
        if valid:
            matchedFilelistList[0].append(filelistList[0][k])
            for i, item in enumerate(item_buffer):
                matchedFilelistList[i + 1].append(item_buffer[i])
        else:
            continue
    if verbose:
        print(' Groups before matching:')
        for i, group in enumerate(filelistList):
            print('   Group {} ({} files): {} ...'.format(i, len(group), group[0]))
        print(' Get {}  files after matching:'.format(len(matchedFilelistList[0])))
        for i in range(len(matchedFilelistList)):
            print(' Group {}'.format(i))
            for item in matchedFilelistList[i]:
                print('   --> {}'.format(item))
        import sys
        sys.stdout.flush()
    return matchedFilelistList


#   用于文件分组
#   例如：有10个slide，有10个mask，10个coord分别构成list，又组在一起作为输入filelistList
#   我的返回结果则是类似于{[[6slide],[6mask],[6coords]],[[4slide],[4mask],[4coords]]}这种构成
#   这个例子中，系列数=3，组数=2，项目数=10
def partition(filelistList, proportion):
    proportion = np.array(proportion)
    proportion = proportion / np.sum(proportion)  # 归一化
    series_amount = len(filelistList)
    group_amount = len(proportion)
    item_amount = len(filelistList[0])
    #   检查每组元素数是否匹配
    item_in_each_group = [len(series) for series in filelistList]  # 统计每个系列的元素数，它们应当是相等的
    if len(np.unique(item_in_each_group)) > 1:
        print(' Items in each group not match ({})'.format(item_in_each_group))
        raise ValueError
    #   获得打乱的下标
    idx = np.arange(item_amount)
    np.random.shuffle(idx)
    divide_prop = np.cumsum(proportion)  # 例如0.5,0.5，会得到0.5,1
    divide_position = (item_amount * divide_prop).astype(np.int)
    divide_position = np.concatenate((np.array([0]), divide_position))
    #   准备结果
    partition_result = []
    for group_id in range(group_amount):
        this_group = []
        select_idx = list(idx[divide_position[group_id]:divide_position[group_id + 1]])
        for series_id in range(series_amount):
            this_group.append([filelistList[series_id][item_idx] for item_idx in select_idx])
        partition_result.append(this_group)
    return partition_result


#   对一个文件路径列表进行过滤，获取纯文件名包含指定字段的文件
def filter_files(filelist, filters, ext=''):
    result = []
    if not isinstance(filters, list) and not isinstance(filters, tuple):
        filters = [filters]
    for item in filelist:
        item_purename = purename(item)
        count_list = [item_purename.count(filter) for filter in filters]
        if np.sum(count_list) > 0:
            #   这说明出现啦
            if (not ext == '') and (not item.endswith(ext)):
                #   如果后缀名被设定了，却不符合条件，那就没办法了
                continue
            #   坚持到最后的王者
            result.append(item)
    return result


#   创建一个文件夹
def create_dir(path):
    if isinstance(path, list):
        for item in path:
            create_dir(item)
    else:
        if path == '':
            return
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except Exception as e:
            print('* Error happens while trying create folder {}, due to {}'.format(path, e))


#   替换文件拓展名,ext为不带英文点的拓展名，例如'tif'
def ext_rep(file_name, ext):
    base_name, _ = os.path.splitext(file_name)
    ext = ext.split('.')[-1]
    return base_name + '.' + ext


#   获取纯文件名
def purename(path):
    return os.path.splitext(os.path.basename(path))[0]


#   找到pinglib的目录（因为可能运行不同文件夹下的程序，所以不一定pinglib总处在根目录中）
def libpath():
    # for i in range(5):
    #     try_path = os.path.join("." * i, "pinglib_sep")
    #     if os.path.isdir(try_path):
    #         return try_path
    # print("Fail to locate pinglib_sep")
    # raise ValueError
    import importlib
    return os.path.dirname(importlib.util.find_spec('pinglib_sep').origin)


'''----------------------以下为废弃的----------------------'''

# '''-------- 更换到slide_sampler之后，Dataset的功能不再是刚需了
#             主要的原因是，Dataset的功能是实现数据集自动划分的，
#             如果你的集合已经划分好了，是不需要这个傻逼东西的    --------------'''
#
# class Dataset():
#     #   在创建时，指定名称，输入、输出个数
#     def __init__(self, file_channels=1, name='default_dataset'):
#         self.name = name
#         self.file_channels = file_channels
#         self.files = [[] for i in range(file_channels)]
#         self.dataset = {}
#         self.dataset_idx = {}
#         self.path = os.path.join('programdata', 'dataset', self.name + '.pkl')
#
#     #   送入文件列表或路径(如果没有，就显式地指定None)
#     def push(self, *args):
#         #   检查送入的变量个数
#         if not len(args) == self.file_channels:
#             msg = '* Error while pushing into dataset: file channel mismatch'
#             return msg
#         #   将变量依次送入列表
#         all_args = []
#         all_args_length = []
#         single_file_mode = False
#         for arg in args:
#             if arg is None:
#                 all_args.append([None])
#                 all_args_length.append(1)
#             elif isinstance(arg, str):
#                 if os.path.isdir(arg):
#                     #   如果是字符串的话，检查是否为文件夹，是的话就加入文件
#                     temp_list = get_file_list(arg)
#                     all_args.append(temp_list)
#                     all_args_length.append(len(temp_list))
#                 elif os.path.isfile(arg):
#                     #   如果是字符串，还有可能就是一个文件
#                     all_args.append([arg])
#                     all_args_length.append(1)
#                     single_file_mode = True
#             elif isinstance(arg, list):
#                 #   还有可能送进来的arg本来就是一个list
#                 all_args.append(arg)
#                 all_args_length.append(len(arg))
#         #   检查数目匹配性
#         larger_than_1 = [x for x in all_args_length if x > 1]
#         if len(np.unique(larger_than_1)) > 1:
#             #   如果有两个不同输入的数目超过了1，说明不匹配
#             msg = '* Error while pushing into dataset: mismatched multiple files amount'
#             return msg
#         max_length = np.max(all_args_length)
#         if single_file_mode and max_length > 1:
#             msg = '* Error while pushing into dataset: single file with multiple files'
#             return msg
#         #   然后正式push
#         for i, this_arg in enumerate(all_args):
#             #   先拓展None元
#             if len(this_arg) == 1:
#                 this_arg = this_arg * max_length
#             #   然后压进去
#             self.files[i] += this_arg
#         msg = '     ' + str(max_length) + ' object pushed into dataset'
#         return msg
#
#     #   对里面的内容进行划分，如划分比例[1,0.3,0.3]，如果你指定名字，就会存在self.datasets['name']中，否则名字默认0,1,2
#     def divide(self, proportion, name_list=None):
#         file_amount = len(self.files[0])
#         idx = np.arange(file_amount)
#         np.random.shuffle(idx)
#         divide_prop = np.cumsum(proportion) / np.sum(proportion)
#         divide_position = (file_amount * divide_prop).astype(np.int)
#         divide_position = np.concatenate((np.array([0]), divide_position))
#         if name_list is None:
#             name_list = [str(i) for i in range(len(proportion))]
#         for i, subset_name in enumerate(name_list):
#             index = idx[divide_position[i]:divide_position[i + 1]].tolist()
#             self.dataset[subset_name] = [[self.files[j][k] for k in index] for j in range(self.file_channels)]
#             self.dataset_idx[subset_name] = index
#         msg = '     ' + ' dataset divided'
#         return msg
#
#     # 按照名字读取
#     def load(self):
#         try:
#             pickle_file = open(self.path, 'rb')
#             exist_dataset = pickle.load(pickle_file)
#             pickle_file.close()
#             self.files = exist_dataset.files
#             self.dataset = exist_dataset.dataset
#             self.dataset_idx = exist_dataset.dataset_idx
#             print('* Dataset loaded from: ' + self.path + ' [Dataset @ files]')
#             return True
#         except:
#             print('* Fatal: Load dataset failed' + ' [Dataset @ files]')
#             return False
#
#     # 保存
#     def save(self):
#         create_dir(os.path.join('programdata', 'dataset'))
#         pickle_file = open(self.path, 'wb')
#         pickle.dump(self, pickle_file)  # 把自己写入pickle文件
#         pickle_file.close()
#         msg = '* Dataset saved at: ' + self.path + ' [Dataset @ files]'
#         return msg


#   这个东西起初是用于自动识别最新模型的，现在在trainer框架中会自动保存最佳和最近模型，这个东西不再需要了
#   获取路径下所有文件并排序(按照文件名中的数字，要求必须有数字)
#   如果想要更复杂的功能，参考：
#   x.split('.')，按照.划分字符串
#   ''.join(x)，把list中的字符串联起来
#   num =''.join([x for x in str if x.isdigit()])
#   filenames.sort(key=lambda x:int(x[:-4])) #  按指定的位置排序
# def list_file_and_sort(path):
#     filenames = os.listdir(path)
#     number_inside = []
#     for filename in filenames:
#         split_list = filename.split('.')  # 按.分割
#         if len(split_list) >= 2:
#             split_list = split_list[:-1]
#             filter_name = ''.join(split_list)
#         number_str = ''.join([x for x in filter_name if x.isdigit()])
#         number_inside.append(int(number_str))
#     index = np.argsort(np.array(number_inside))
#     new_filenames = []
#     for i in index:
#         new_filenames.append(filenames[i])
#     return new_filenames


'''-------------这个Dataset只能用单输入单输出的，被新的Dataset取代了-----------------'''
# class Dataset():
#     def __init__(self, name='default_dataset'):
#
#         self.name = name
#         self.path = os.path.join('programdata', 'dataset', self.name + '.pkl')
#         self.slide_paths = []
#         self.ref_paths = []
#         self.dataset = {}
#         pass
#
#     #   可以送入路径（一次送一对文件夹，不要送list），也可以送入文件列表（list）
#     def push(self, slide_paths, ref_paths=None):
#         if isinstance(slide_paths, str):
#             slide_paths = get_file_list(slide_paths)
#         if isinstance(ref_paths, str):
#             ref_paths = get_file_list(ref_paths)
#         if (ref_paths is None) and (isinstance(slide_paths, list)):
#             ref_paths = [None] * len(slide_paths)
#         self.slide_paths = self.slide_paths + slide_paths
#         self.ref_paths = self.ref_paths + ref_paths
#
#     #   对里面的内容进行划分，如划分比例[1,0.3,0.3]，如果你指定名字，就会存在self.datasets['name']中，否则名字默认0,1,2
#     def divide(self, proportion, name_list=None):
#         slide_amount = len(self.slide_paths)
#         idx = np.arange(slide_amount)
#         np.random.shuffle(idx)
#         divide_prop = np.cumsum(proportion) / np.sum(proportion)
#         divide_position = (slide_amount * divide_prop).astype(np.int)
#         divide_position = np.concatenate((np.array([0]), divide_position))
#         if name_list is None:
#             name_list = [str(i) for i in range(len(proportion))]
#         for i, subset_name in enumerate(name_list):
#             index = idx[divide_position[i]:divide_position[i + 1]].tolist()
#             self.dataset[subset_name] = {}
#             self.dataset[subset_name]['x'] = [self.slide_paths[j] for j in index]
#             self.dataset[subset_name]['y'] = [self.ref_paths[j] for j in index]
#
#     # 按照名字读取
#     def load(self):
#         try:
#             pickle_file = open(self.path, 'rb')
#             exist_dataset = pickle.load(pickle_file)
#             pickle_file.close()
#             self.slide_paths = exist_dataset.slide_paths
#             self.ref_paths = exist_dataset.ref_paths
#             self.dataset = exist_dataset.dataset
#             print('* Dataset loaded from: ' + self.path + ' [Dataset @ files]')
#         except:
#             print('* Fatal: Load dataset failed' + ' [Dataset @ files]')
#             raise (RuntimeError)
#
#     # 保存
#     def save(self):
#         create_dir(os.path.join('programdata', 'dataset'))
#         pickle_file = open(self.path, 'wb')
#         pickle.dump(self, pickle_file)  # 把自己写入pickle文件
#         pickle_file.close()
#         print('* Dataset saved at: ' + self.path + ' [Dataset @ files]')
