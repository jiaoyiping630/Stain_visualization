#!/usr/bin/env python
# -*- coding: utf-8 -*-
#   Monitor主要实现对任务的监控和管理
import os, sys, time
import pickle
from traceback import format_exc


#   记录日志
#   当需要记录时，使用logger.debug(msg)即可
def Logger(path=None):
    from .files import create_dir
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    if path is None or path == '':
        path = 'default_logger.log'
    file_formatter = logging.Formatter('%(asctime)s [line:%(lineno)d @ %(filename)s]: %(message)s')
    command_formatter = logging.Formatter('%(message)s')

    if not os.path.isdir(os.path.dirname(path)):
        create_dir(os.path.dirname(path))
        time.sleep(10)  # wait a while, to make sure the folder created...

    try:
        file_handle = logging.FileHandler(path, mode='w')
    except:
        print('Exception raised due to {}'.format(format_exc()))
        print('Do we have the folder {} ? --> {}'.format(os.path.dirname(path),
                                                         os.path.isdir(os.path.dirname(path))))
        print('Well lets just print the result.')
        return _backup_logger()

    file_handle.setLevel(logging.DEBUG)
    file_handle.setFormatter(file_formatter)

    command_handle = logging.StreamHandler()
    command_handle.setLevel(logging.INFO)
    command_handle.setFormatter(command_formatter)

    logger.addHandler(file_handle)
    logger.addHandler(command_handle)
    return logger


#   因为Logger频繁的路径不存在，我已经无法忍受了，这个备胎的功能就是直接print！
class _backup_logger():
    def __init__(self):
        pass

    def info(self, message):
        print(message)
        sys.stdout.flush()

    def debug(self, message):
        print(message)
        sys.stdout.flush()


#   在终端显示进度条
def terminal_viewer(current, total, head="Percent: ", tail="", interval=0):
    import sys
    bar_length = 60
    percent = current / total
    hashes = '#' * int(percent * bar_length)
    spaces = ' ' * (bar_length - len(hashes))
    if interval == 0 or int(current * 100 / total) % interval == 0:
        sys.stdout.write("\r%s[%s] %d%% %d/%d %s"
                         % (head, hashes + spaces, percent * 100, current, total, tail))
        sys.stdout.flush()


#   给自己的邮箱发邮件，可带附件
def mail_to_me(filelist=None):
    try:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.header import Header
        from email.mime.text import MIMEText
        from email.mime.application import MIMEApplication
        import os
        _user = "feilab88@126.com"
        _pwd = "qwerzxcv1234"
        _to = "ping@seu.edu.cn"
        #   使用MIMEText构造符合smtp协议的header及body
        msg = MIMEMultipart()
        msg["Subject"] = "お父さん、情報を確認してください"
        msg["From"] = _user
        msg["To"] = _to
        part = MIMEText("ラボのプログラムが実行されました。 チェックしてください。", 'plain', 'utf-8')
        msg.attach(part)
        #   添加附件
        for each_file in filelist:
            if os.path.isfile(each_file):
                basename = os.path.basename(each_file)
                part = MIMEApplication(open(each_file, 'rb').read())
                part.add_header('Content-Disposition', 'attachment', filename=basename)
                msg.attach(part)
        # 登录和发送
        s = smtplib.SMTP("smtp.126.com", timeout=30)  # 连接smtp邮件服务器,端口默认是25
        s.login(_user, _pwd)  # 登陆服务器
        s.sendmail(_user, _to, msg.as_string())  # 发送邮件
        s.close()
    except:
        print('* Error while sending email. ' + '[sent_to_me @ monitor]')


'''------------不常用的内容----------------'''


#   实现中间结果的记录
class Recorder():
    def __init__(self, path=None, override=False):
        #   如果没有指明名字，那么将设置为'default_recorder'，并且默认是重写的
        if path is None or path == '':
            path = 'default_recorder.pkl'
        self.path = path
        self.data = {}
        #   如果配置为重写，或者没有找到这个文件，那么就
        if override or (not os.path.isfile(self.path)):
            #   1. 尝试把它删了
            try:
                os.remove(self.path)
            except:
                pass
            #   2. 把自己先存起来再说
            self.save()
        else:
            self.load()

    #   把数据推入recorder，因为保存可能耗时较多，默认不保存，也可以置save为True立刻保存
    #   输入的data是你需要保存的数据
    #   field是你要保存到recorder的字段，为字符串
    #   如果field是list，则data也需要为同长度的list，以便对应存放
    def push(self, data, field, save=False, replace=False):
        if type(field) == list:
            for each_data, each_field in zip(data, field):
                if (each_field not in self.data.keys() or replace):
                    self.data[each_field] = []  # 如果不存在字段，需要初始化
                self.data[each_field].append(each_data)  # 否则直接append
        else:
            if (field not in self.data.keys() or replace):
                self.data[field] = []
            self.data[field].append(data)
        if save:
            self.save()

    def load(self):
        pickle_file = open(self.path, 'rb')
        exist_recorder = pickle.load(pickle_file)
        pickle_file.close()
        self.data = exist_recorder.data

    #   把当前的Recorder保存下来
    def save(self):
        pickle_file = open(self.path, 'wb')
        pickle.dump(self, pickle_file)  # 把自己写入pickle文件
        pickle_file.close()

    #   dig方法按照指定的层级关系访问数据，并进行聚合，以数组的方式呈递出来
    #   如fields_list=['loss']表示对loss的数据串联起来
    #   fields_list=['patch_info','target_class']则可以获取采的样本的类别统计
    #   如果你配置好了索引，里面也可以掺数字
    def dig(self, field_list):
        import numpy as np
        if isinstance(field_list, str):
            field_list = [field_list]
        data = self.data
        output = []
        for field in field_list:
            data = data[field]
        for item in data:
            output.append(item)
        try:
            return np.array(output)
        except:
            return output
