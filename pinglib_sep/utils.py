#!/usr/bin/env python
# -*- coding: utf-8 -*-
import timeit
import os, platform, sys, csv
import numpy as np
from tqdm import tqdm
from .files import create_dir

'''--------------老牌的常用方法-------------------------'''


#   提供tic()和toc()，模仿Matlab计时
class Timer():
    #   这里改用了timeit模块，具有跨平台精度
    #   time.clock是处理器时间，time.time是精确时间
    def __init__(self):
        self.time = None

    def tic(self):
        self.time = timeit.default_timer()

    def toc(self, verbose=False, restart=False):
        elapse = (timeit.default_timer() - self.time)
        if verbose:
            print(' time elapsed : {} s'.format(elapse))
        if restart:
            self.tic()
        return elapse


#   把数据存在指定的pkl文件中
def save_variables(var_list, target_path, override=True):
    from .files import create_dir
    import pickle
    import os
    #   若文件存在且不覆写，直接返回
    if os.path.isfile(target_path) and not override:
        return
    #   如果目标路径的文件夹不存在，先创建
    try:
        folder_path, _ = os.path.split(target_path)
        create_dir(folder_path)
    except:
        pass
    #   然后保存数据
    if not isinstance(var_list, list):
        var_list = [var_list]
    pickle_file = open(target_path, 'wb')
    for item in var_list:
        pickle.dump(item, pickle_file)
    pickle_file.close()


#   从指定的pkl文件中读取数据
def load_variables(target_path):
    import pickle
    return_list = []
    pickle_file = open(target_path, 'rb')
    while True:
        try:
            item = pickle.load(pickle_file)
            return_list.append(item)
        except:
            break
    pickle_file.close()
    return return_list


#   重命名文件
#   ASAP中，mask最好是用_likelihood_map这样的后缀，可以自动载入
def rename(folder_path, ext='', find_str='', replace_str='', new_prefix='', new_suffix='', new_ext=None,
           ignore_error=False,
           verbose=True):
    import os
    from .files import get_file_list, purename
    file_list = get_file_list(folder_path, ext=ext)
    #   如果新的拓展名不为None，那么需要替换拓展名，这里如果发现没有点，要加上点
    if new_ext is not None:
        if not new_ext.startswith('.'):
            new_ext = '.' + new_ext
    #   为每个文件重命名
    counter = 0
    for file_path in file_list:
        pure_name = purename(file_path)
        _, ext = os.path.splitext(file_path)
        new_pure_name = new_prefix + pure_name.replace(find_str, replace_str) + new_suffix
        if new_ext is None:
            new_name = os.path.join(folder_path, new_pure_name + ext)
        else:
            new_name = os.path.join(folder_path, new_pure_name + new_ext)
        if file_path != new_name:
            if ignore_error:
                try:
                    os.rename(file_path, new_name)
                    if verbose:
                        print('  Renamed {} --> {}'.format(file_path, new_name))
                    counter += 1
                except Exception as e:
                    print('  Error while renaming {} --> {}'.format(file_path, new_name))
            else:
                os.rename(file_path, new_name)
                if verbose:
                    print('  Renamed {} --> {}'.format(file_path, new_name))
                counter += 1
    if verbose:
        print('  Finish renaming {} items.'.format(counter))


#   新的拷贝程序，如果出现文件夹冲突，可以自动建立新的，避免覆写
def copy_to_dir(filelist, target_folder, force_new_folder=False, verbose=False):
    import os
    #   根据是否要求必须写入新文件夹中，生成目标目录
    counter = 0
    appendix = ''
    while True:
        #   首先判断目标文件夹是否存在
        if os.path.isdir(target_folder + appendix):
            #   如果已经存在了，而且要求必须是新文件夹，那只能尝试下一个了
            if force_new_folder:
                appendix = '_' + str(counter)
                counter += 1
            else:
                #   如果已存在，且并不要求必须是全新文件夹，那么这样就行啦
                target_folder += appendix
                break
        else:
            #   如果不存在，那你他妈还不赶紧创建一个？
            from .files import create_dir
            create_dir(target_folder)
    #   进行拷贝
    import shutil, os
    import numpy as np
    file_amount = len(filelist)
    file_sizes = [round(os.path.getsize(this_file) / float(1024 * 1024), 2) for this_file in filelist]
    total_size = np.sum(file_sizes)
    timer = Timer()
    timer.tic()
    time_elapse = 0
    size_elapse = 0
    filelist_dest = []
    for i, this_file in enumerate(filelist):
        #   如果遇到None，则不拷贝，但也将None记录到结果内
        if this_file is None:
            filelist_dest.append(None)
            continue
        #   正常拷贝文件
        file_dest = os.path.join(target_folder, os.path.basename(this_file))
        filelist_dest.append(file_dest)
        if os.path.isfile(file_dest):
            print('File {} already exist, skipped'.format(file_dest))
            sys.stdout.flush()
            continue
        if verbose:
            print('Copying {}/{}: {} ...'.format(i + 1, file_amount, this_file))
            sys.stdout.flush()
        try:
            shutil.copy(this_file, file_dest)
        except Exception as e:
            print('   Fail to copy {} using shutil due to {}'.format(this_file, e))
            sys.stdout.flush()
            if platform.system() == "Windows":
                flag = os.system("copy {} {}".format(this_file, file_dest))
            else:
                flag = os.system("cp {} {}".format(this_file, file_dest))
            if flag == 1:
                print('   Fail to copy using subshell cp/copy')
                sys.stdout.flush()
        finally:
            if verbose:
                time_delta = timer.toc(restart=True)
                size_elapse += file_sizes[i]
                time_elapse += time_delta
                print('   Size elapse: {}/{} ({}%), ETA:{}'.
                      format(round(size_elapse, 2),
                             round(total_size, 2),
                             round(size_elapse * 100 / total_size, 2),
                             round(time_elapse * total_size / size_elapse - time_elapse, 2)
                             ))

    if verbose:
        print('Copying to local done, time elapse = {}'.format(time_elapse))
        sys.stdout.flush()
    return filelist_dest


#   一个更短小的shutil.copy的替代品（适用于单文件）
def shutilcopy(this_file, file_dest):
    import platform
    if platform.system() == "Windows":
        flag = os.system("copy {} {}".format(this_file, file_dest))
    else:
        flag = os.system("cp {} {}".format(this_file, file_dest))
    if flag == 1:
        print('Fail to copy using subshell cp/copy')
        sys.stdout.flush()
        raise ValueError


#   用于将文件分别拷贝到对应文件的目录下
#   该功能常用于，例如将prediction文件分别移动到TCGA原片所在的目录下
def distribute_copy(files_to_be_copied, reference_paths):
    file_paths = files_to_be_copied
    if not len(file_paths) == len(reference_paths):
        print('File list length not match ({} file to be copied, {} file to be referenced)'.format(
            len(file_paths), len(reference_paths)))
        raise ValueError
    print('Coping files ...')
    for (file_path, reference_path) in tqdm(zip(file_paths, reference_paths)):
        basename = os.path.basename(file_path)
        [dirname, _] = os.path.split(reference_path)
        target_path = os.path.join(dirname, basename)
        if os.path.isfile(target_path):
            print('File {} exist, skipped'.format(target_path))
        shutilcopy(file_path, target_path)


#   用于解析从命令行传递来的参数，获取partition设定（以便批量开启多任务）
#   好像也可以通过sys.argv来实现，这样就不需要在python的时候写-c, -t了……
def parse_partition():
    import argparse
    argument_parser = argparse.ArgumentParser(description='Partition setting')
    argument_parser.add_argument('-c', '--current', default=1, type=int, help='current partition')
    argument_parser.add_argument('-t', '--total', default=1, type=int, help='total partitions')
    argument_parser.add_argument('-g', '--gpu', default=0, type=int, help='gpu id')

    arguments = vars(argument_parser.parse_args())
    partition = (arguments['current'], arguments['total'])
    partition_min = min(partition)
    partition_max = max(partition)
    gpu_id = arguments['gpu']

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print('Parsed partition setting = {}, using gpu {}'.format((partition_min, partition_max), gpu_id))
    sys.stdout.flush()
    return (partition_min, partition_max)


def filter_by_partition(series, partition):
    #   如果发现给的东西不是[[]]的形式，则是以单个list传入的，并注意输出也符合同样的习惯
    if not isinstance(series[0], list):
        series = [series]
        single_flag = True
    else:
        single_flag = False

    original_number = len(series[0])
    partition_max = max(partition)
    partition_min = min(partition) % partition_max

    filtered_series = [item[partition_min::partition_max] for item in series]
    filtered_number = len(filtered_series[0])
    print('{} out of {} groups were filtered due to partition setting {}'.format(filtered_number, original_number,
                                                                                 partition))
    if single_flag:
        return filtered_series[0]
    else:
        return filtered_series


def gpu_config(gpu_id):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def thread_config(thread_num):
    import os
    os.environ["OMP_NUM_THREADS"] = str(thread_num)


'''---------------针对具体类型文件的一些工具-------------'''


#   专门用于写Excel工作表的工具
class WorkBookwriter():
    def __init__(self, target_path, worksheet_name=None, col_size=99999):
        import xlwt
        self.target_path = target_path
        self.workbook = xlwt.Workbook(encoding='ascii')
        self.sheet_infos = {}
        self.current_worksheet = None
        self.current_worksheet_name = None

        #   初始化的时候，直接建一个sheet先
        worksheet_name = 'default' if worksheet_name is None else worksheet_name
        self.sheet(worksheet_name)

        self.col_size = col_size

    #   跳转至指定sheet，并保存当前状态
    def sheet(self, worksheet_name):
        #   首先保存当前的状态
        if self.current_worksheet is not None:
            self.sheet_infos[self.current_worksheet_name]['sheet'] = self.current_worksheet
            self.sheet_infos[self.current_worksheet_name]['row'] = self.current_row
            self.sheet_infos[self.current_worksheet_name]['col'] = self.current_col
        if worksheet_name not in self.sheet_infos.keys():
            #   如果还没有这个sheet，则需要新建
            current_sheet = self.workbook.add_sheet(worksheet_name)
            self.current_worksheet = current_sheet
            self.current_worksheet_name = worksheet_name
            self.current_row = 1  # 这里的行列号和excel的习惯一致，从1开始
            self.current_col = 1

            self.sheet_infos[worksheet_name] = {}
            self.sheet_infos[worksheet_name]['sheet'] = current_sheet
            self.sheet_infos[worksheet_name]['row'] = self.current_row
            self.sheet_infos[worksheet_name]['col'] = self.current_col
        else:
            #   如果已经有这个sheet了，则读取信息
            self.current_worksheet = self.sheet_infos[worksheet_name]['sheet']
            self.current_worksheet_name = worksheet_name
            self.current_row = self.sheet_infos[worksheet_name]['row']
            self.current_col = self.sheet_infos[worksheet_name]['col']

    # def new_sheet(self, worksheet_name):
    #     self.worksheet = self.workbook.add_sheet(worksheet_name)
    #     self.current_row = 1  # 这里的行列号和excel的习惯一致，从1开始
    #     self.current_col = 1

    def set_col(self, value):
        self.col_size = value

    def set_current_position(self, row, col):
        self.current_row = row
        self.current_col = col

    def go_to_next_row(self):
        self.current_row = self.current_row + 1
        self.current_col = 1

    #   写一个数据，并向右移动位置，若达到设定列则自动换行
    def write(self, data):
        if isinstance(data, list) or isinstance(data, tuple):
            #   若对象是一个list，则依次写入每一个元素
            for item in data:
                self.write(item)
        elif isinstance(data, np.ndarray):
            #   若对象是一个numpy数组，则将其转换为list后写入
            self.write(data.tolist())
        else:
            #   在写之前，判断是不是要换行
            if self.current_col > self.col_size:
                self.current_col = 1
                self.current_row = self.current_row + 1
            try:
                self.current_worksheet.write(self.current_row - 1, self.current_col - 1, label=data)
            except:
                self.current_worksheet.write(self.current_row - 1, self.current_col - 1, label=str(data))
            #   写完之后，增加列号
            self.current_col = self.current_col + 1

    def writerow(self, data):
        self.write(data)
        self.go_to_next_row()

    #   向当前位置（或先设定位置）写一个数据，并向右移动位置(不换行)
    def write_and_move_right(self, data, row=None, col=None):
        if row is not None:
            self.current_row = row
        if col is not None:
            self.current_col = col
        self.current_worksheet.write(self.current_row - 1, self.current_col - 1, label=data)

    def finish(self):
        self.workbook.save(self.target_path)


#   从csv中读取数据(第一行是标题)
class Csv_data():
    def __init__(self, csv_path):
        import csv
        csv_file = open(csv_path, 'r', encoding='utf-8')
        reader = csv.reader(csv_file)

        self.keys = None
        self.values = None
        self.item_amount = 0
        self.series_amount = 0

        for i, item in enumerate(reader):
            if i == 0:
                self.keys = item
                self.series_amount = len(self.keys)
                self.values = [[] for _ in range(self.series_amount)]
            else:
                for j, subitem in enumerate(item):
                    self.values[j].append(subitem)
                self.item_amount += 1

        csv_file.close()

    def get(self, key, type=None):
        if self.keys is None or self.values is None:
            print('Error: No data available')
            raise ValueError
        if key not in self.keys:
            print('Error: Key ''{}'' not founded in csv data ({})'.format(key, self.keys))
            raise ValueError
        index = self.keys.index(key)
        data = self.values[index]
        if type == 'float' or 'int' or 'double':
            return [float(item) for item in data]
        else:
            return data


class CSV_writer():
    def __init__(self, csv_path):
        from .files import create_dir
        folder_path, _ = os.path.split(csv_path)
        create_dir(folder_path)
        self.csv_path = csv_path

    def write(self, content):
        if not isinstance(content, list):
            if isinstance(content, np.ndarray):
                content = content.tolist()
            else:
                content = list(content)
        csv_file = open(self.csv_path, 'a', newline='')
        csv_writer = csv.writer(csv_file, dialect='excel')
        csv_writer.writerow(content)
        csv_file.close()


#   专用于内存占用分析
class Memory_diagnosis():
    def __init__(self, csv_path):
        from .files import create_dir
        folder_path, _ = os.path.split(csv_path)
        create_dir(folder_path)
        self.csv_path = csv_path

        csv_file = open(self.csv_path, 'a', newline='')
        csv_writer = csv.writer(csv_file, dialect='excel')
        csv_writer.writerow(['Time', 'Occupy', 'Log'])
        csv_file.close()

        self.timer = Timer()
        self.timer.tic()

    def record(self, log=''):
        import psutil
        occupy = round(psutil.virtual_memory().used / (1024 ** 2), 2)

        csv_file = open(self.csv_path, 'a', newline='')
        csv_writer = csv.writer(csv_file, dialect='excel')
        csv_writer.writerow([self.timer.toc(), occupy, log])
        csv_file.close()

    # annotation的方法
    # isClockwise()
    # getNumberOfPoints()
    # getType()
    # getImageBoundingBox
    # getLocalBoundingBox
    # getCenter
    # getCoordinate得到的是point
    # getCoordinates
    #
    # AnnotationList的方法：
    # addGroup
    # addAnnotation
    # getGroup
    # getAnnotation
    # getAnnotations
    # getGroups
    #
    # AnnotationService的方法：
    # getList
    # getRepository
    # saveRepositoryToFile
    #
    # AnnotationToMask的方法：
    # convert
    #
    # Repository的方法：
    # setSource
    # load
    # save
    #
    # Point的方法：
    # getX
    # getY

    '''---------已废弃，被Dataset类代替--------------'''
    # # 将所有的数据进行划分
    # #   slide_list：所有的slide的列表
    # #   mask_list：所有的mask的列表
    # #   name_list：待划分的集合名，如['train','valid','test']
    # #   proportion_list：在几个集合中的样本数分配比例，如[0.7,0.3,0.3]
    # def dataset_divide(slide_list, mask_list, name_list, proportion_list):
    #     import numpy as np
    #     slide_amount = len(slide_list)
    #     idx = np.arange(slide_amount)
    #     np.random.shuffle(idx)
    #     datasets = []
    #     divide_prop = np.cumsum(proportion_list) / np.sum(proportion_list)
    #     divide_position = (slide_amount * divide_prop).astype(np.int)
    #     divide_position = np.concatenate((np.array([0]), divide_position))
    #     for i, subset_name in enumerate(name_list):
    #         this_dataset = {}
    #         this_dataset['name'] = subset_name
    #         index = idx[divide_position[i]:divide_position[i + 1]].tolist()
    #         this_dataset['index'] = index
    #         this_dataset['slide_paths'] = []
    #         this_dataset['mask_paths'] = []
    #         for j in index:
    #             this_dataset['slide_paths'].append(slide_list[j])
    #             this_dataset['mask_paths'].append(mask_list[j])
    #         datasets.append(this_dataset)
    #     return datasets


#   解压tar文件
def tar_extract(tar_path, extract_path):
    import tarfile
    tar = tarfile.open(tar_path)
    names = tar.getnames()
    for name in names:
        try:
            tar.extract(name, path=extract_path)
        except Exception as e:
            print('Error happen while dealing with: {}, due to {}'.format(name, e))
    tar.close()


'''---------------以下是不太常用的功能了----------------'''


#   获取调用参数有几个输出，可以实现matlab的nargout类似的功能(不完美，调用它的语句必须写在一行内)
def nargout(*args):
    import traceback
    callInfo = traceback.extract_stack()
    callLine = str(callInfo[-3].line)
    split_equal = callLine.split('=')
    # no_equal = len(split_equal) == 1
    split_comma = split_equal[0].split(',')
    # num = len(args) if no_equal else len(split_comma)
    num = len(split_comma)
    return args[0:num] if num > 1 else args[0]


#   将一个字典里的值输出为文本文档
def dict_to_txt(dict_obj, txt_path='temp.txt'):
    f = open(txt_path, 'a')
    for i in dict_obj.keys():
        f.write(i)
        f.write(':\t\t')
        f.write(str(dict_obj[i]))
        f.write('\n')
        print(i + ' : ' + str(dict_obj[i]))


#   把一个pickle对象里的所有属性-值拷贝到自己里面
def load_from_obj(self, obj_pickle):
    import pickle
    pickle_file = open(obj_pickle, 'rb')
    exist_obj = pickle.load(pickle_file)
    pickle_file.close()

    for item_name in exist_obj.__dict__:
        exec('self.' + item_name + '=exist_obj.' + item_name)


#   进行训练集、验证集自动切分
#   list: 由list构成的list，各个list应等长
#   train_prop: 训练集的比例
#   record_path: 记录文件，如果非None，则会把切分的结果保存在这个文件里（csv格式）
#   seed: 随机切分时的随机数种子
def train_valid_split(lists, train_prop=0.7, record_path=None, seed=0):
    modality_num = len(lists)
    item_num = len(lists[0])
    #   检验各列表是否等长
    for i in range(modality_num):
        assert len(lists[i]) == item_num
    #   获取随机索引
    np.random.seed(seed)
    train_num = int(item_num * train_prop)
    train_index = np.random.choice(item_num, train_num, replace=False).tolist()
    valid_index = [idx for idx in range(item_num) if idx not in train_index]
    #   获取切分的数据集
    train_set = []
    valid_set = []
    for i in range(modality_num):
        train_set.append([lists[i][idx] for idx in train_index])
    for i in range(modality_num):
        valid_set.append([lists[i][idx] for idx in valid_index])
    #   存储切分结果
    if record_path is not None:
        folder_path, _ = os.path.split(record_path)
        create_dir(folder_path)
        csv_file = open(record_path, 'a', newline='')
        csv_writer = csv.writer(csv_file, dialect='excel')
        csv_writer.writerow(['raw_idx', 'dataset'] + ['Modality_{}'.format(i + 1) for i in range(modality_num)])
        for j in range(item_num):
            this_row_content = [str(j)]
            if j in train_index:
                this_row_content += ['train']
            else:
                this_row_content += ['valid']
            this_row_content += [lists[i][j] for i in range(modality_num)]
            csv_writer.writerow(this_row_content)
        csv_file.close()

    return train_set, valid_set


'''---------------------这几个文件拷贝方法已作废----------------------'''

# #   把一系列文件拷贝到deadpool上，并返回在deadpool上对应文件的路径
# def copy_to_local(filelist, savedir='/home/user/input', show_process=False):
#     from .files import create_dir
#     create_dir(savedir)
#     import shutil, os
#     filelist_dest = []
#     for this_file in filelist:
#         file_dest = os.path.join(savedir, os.path.basename(this_file))
#         filelist_dest.append(file_dest)
#         if os.path.isfile(file_dest):
#             print('File {} already exist, skipped'.format(file_dest))
#             continue
#         if show_process:
#             print('Copying {}'.format(this_file))
#         shutil.copy(this_file, file_dest)
#
#     if show_process:
#         print('Copying to local done.')
#     return filelist_dest
# #   从某个路径拷单个文件过来到programdata
# def copy_single_file_to_cluster(source_path, target_path):
#     import shutil
#     try:
#         shutil.copy(source_path, target_path)
#         return True
#     except:
#         print('* Error happen whiling copying file')
#         return False
# #   从某个路径拷单个文件过来到默认目录，并返回路径
# def copy_single_file_to_cluster_return_path(source_path):
#     import shutil, os
#     try:
#         target_folder = '/home/user/source/local_temp'
#         target_path = os.path.join(target_folder, os.path.basename(source_path))
#         shutil.copy(source_path, target_path)
#         return target_path
#     except:
#         print('* Error happen whiling copying file')
#         return None
