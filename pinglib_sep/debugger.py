#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os


#   将变量保存为.mat文件，存储于当前文件夹下，方便使用MATLAB观察调试
#   输入：
#       dataList：由待保存变量构成的list
#       nameList：保存变量的命名，最方便的可以直接拷贝为字符串，如[a,b,c]→'[a,b,c]'
#       save_file_name：保存文件的名字，默认为"line_" + 调用行号
#       logging：保存变量时是否在命令行提示
def watch(dataList, nameList=None, save_file_name="", logging=True, with_line_num=False, override=True):
    import sys
    import scipy.io as scio
    if save_file_name == "":
        with_line_num = True
    try:
        #   如果是元组形式给出的，先转换为list
        if type(dataList) is tuple:
            dataList = list(dataList)
        if type(nameList) is tuple:
            nameList = list(nameList)
        if type(dataList) is list:
            #   如果dataList参数是list，那么视后一个参数而定
            if nameList is None:
                #   如果未提供变量名，则生成默认名字
                finalDict = dict()
                i = 0
                for eachData in dataList:
                    i = i + 1
                    finalDict['var' + str(i)] = eachData
            else:
                if type(nameList) is list:
                    #   把变量与名字配对
                    finalDict = dict(zip(nameList, dataList))
                else:
                    #   认为nameList按照字符串给出
                    nameList = nameList.replace('(', '')
                    nameList = nameList.replace(')', '')
                    nameList = nameList.replace('[', '')
                    nameList = nameList.replace(']', '')
                    nameList = nameList.replace(' ', '')
                    nameList = nameList.split(',')
                    finalDict = dict(zip(nameList, dataList))
        else:
            #   如果dataList参数不是list，看看是不是dict形式
            if type(dataList) is dict:
                #   如果是用户组好的dict
                finalDict = dataList
            else:
                #   如果不是dict，把它视为单个变量，直接保存
                if nameList is None:
                    finalDict = {'var': dataList}
                else:
                    finalDict = {nameList: dataList}
        callLine = sys._getframe().f_back.f_lineno
        if save_file_name.endswith('.mat'):
            saveFilePath = save_file_name
        else:
            if with_line_num:
                saveFilePath = save_file_name + '@line_' + str(callLine) + '.mat'
            else:
                saveFilePath = save_file_name + '.mat'
            if not override and os.path.isfile(saveFilePath):
                return
        scio.savemat(saveFilePath, finalDict)
        if logging:
            print('     data saved at line', callLine)
    except Exception as e:
        print('* Error happens while saving variables: {}'.format(e))


#   保存路径的需求：
#       如果是一个完整的路径
def debug(variable_list, override=False):
    import sys, platform
    if platform.system() == "Windows":
        default_path = r"\\deadpool.umcn.nl\pathology\users\yiping\debugging"
    else:
        default_path = '/mnt/synology/pathology/users/yiping/debugging'
    callLine = sys._getframe().f_back.f_lineno
    funName = sys._getframe().f_back.f_code.co_name
    fileName = sys._getframe().f_back.f_code.co_filename
    [_, fileName] = os.path.split(fileName)
    target_path = os.path.join(default_path, '[{}]@[{}][line_{}].pkl'.format(funName, fileName, callLine))
    if os.path.exists(target_path) and not override:
        return
    else:
        from .utils import save_variables
        save_variables(variable_list, target_path=target_path, override=override)


#   该函数用于从pytorch model中抽取各层的权重，计算其统计信息
def get_weight_statistic_pytorch(model):
    info = {}
    max_list = []
    min_list = []
    for i, param in enumerate(model.named_parameters()):
        item = {}
        item['layer_name'] = param[0]
        try:
            weight = param[1].data.numpy()
        except:
            weight = param[1].data.cpu().numpy()
        item['shape'] = weight.shape
        item['mean'] = np.mean(weight)
        item['std'] = np.std(weight)
        item['max'] = np.max(weight)
        item['min'] = np.min(weight)
        max_list.append(item['max'])
        min_list.append(item['min'])
        info[i] = item
    info['max'] = np.max(max_list)
    info['min'] = np.max(min_list)
    return info


#   该函数用于从keras model中抽取各层的权重，计算其统计信息
def get_weight_statistic_keras(model):
    info = {}
    max_list = []
    min_list = []
    for i, layer in enumerate(model.layers):
        item = {}
        item['layer_name'] = layer.name
        weights = layer.get_weights()
        item['amount'] = len(weights)
        shape = []
        mean = []
        std = []
        max = []
        min = []
        for weight in weights:
            shape.append(weight.shape)
            mean.append(np.mean(weight))
            std.append(np.std(weight))
            max.append(np.max(weight))
            min.append(np.min(weight))
            max_list.append(max[-1])
            min_list.append(min[-1])
        item['shape'] = shape
        item['mean'] = mean
        item['std'] = std
        item['max'] = max
        item['min'] = min
        info[i] = item
    info['max'] = np.max(max_list)
    info['min'] = np.max(min_list)
    return info
