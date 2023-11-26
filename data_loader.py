import os
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F


def read_files(file_pattern):
    """
    读取多个Excel和CSV文件，并将它们合并成一个DataFrame对象。
    :param file_pattern: 文件名模式，例如 "*.csv" 或 "*.xlsx"。
    :return: 一个包含所有文件内容的DataFrame对象。
    """
    sheets = ['20','21','22','23','24','25','库水位']
    data_total = np.zeros((0, 7))
    data_temp= np.zeros((31*12, 7))
    df = pd.DataFrame() # 创建空的DataFrame对象
    for file in os.listdir(os.getcwd()+'\\data'): # 遍历当前工作目录下的所有文件
        if file.endswith(file_pattern): # 如果文件名符合文件名模式
            file_path = os.path.join(os.getcwd()+'\\data', file) # 获取文件的完整路径
            try:
                if file.endswith('.xlsx'): # 如果文件是Excel文件
                    sheet_num = 0
                    for sheet in sheets:
                        df = pd.read_excel(file_path, sheet_name=sheet) # 使用pandas库的read_excel类读取Excel文件
                        selected_data = df.iloc[1:32,1:]
                        selected_data = np.array(selected_data)
                        selected_data = selected_data.reshape(-1,1, order='F') # 31*12列向量
                        data_temp[:,sheet_num] = selected_data[:,0]
                        sheet_num += 1

                    data_total = np.concatenate((data_total,data_temp),axis=0) #按照行数增长的方式拼接
            except Exception as e: # 如果读取文件失败
                print(f"Failed to read file '{file_path}': {e}") # 打印错误信息
                continue # 继续查找下一个文件
    return data_total # 返回合并后的DataFrame对象




def dataloading(file_pattern = '.xlsx'):
    data_total = read_files(file_pattern) # 读取所有符合文件名模式的xlsx文件并将其合并成一个data_total对象
    #位置编码

    while ((data_total < 60) | (data_total > 200)).any:
    # 大于200的元素除以10，NaN除外
        mask_200 = data_total > 200
        data_total[mask_200] /= 10
        print('改')
        # 小于60的元素乘以10，NaN除外
        mask_60 = data_total < 60
        data_total[mask_60] *= 10

        tempdata = data_total[~np.isnan(data_total)]
        if tempdata.max()<200 and tempdata.min()>60:
            break


    mon = np.repeat(np.arange(1, 13)[:, np.newaxis], 31, axis=0)
    mon = np.repeat(mon, 8, axis=1)
    mon = mon.reshape(-1,1, order='F')
    day = np.repeat(np.arange(1, 32)[:, np.newaxis], 12, axis=1)
    day = day.reshape(-1,1, order='F') # 31*12列向量
    day = np.repeat(day, 8, axis=1)
    day = day.reshape(-1,1, order='F') # 31*12列向量

    data_total = np.concatenate((mon,day,data_total),axis=1)
    data_total

    rows_with_nan = np.any(np.isnan(data_total), axis=1)
    # 删除包含 NaN 的行
    data_total = data_total[~rows_with_nan]

    data_total = data_total.astype('float32')
    np.savetxt('./data_total.txt',data_total)
    return data_total


class dataset(torch.utils.data.Dataset):
    def __init__(self,data):
        self.data = data
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self,i):
        
        #X=torch.FloatTensor((self.data[i,0:8],))
        X=self.data[i,0:8]
        # 这里注意也是需要转换成tensor的，否则训练会报类型错误
        y=self.data[i,-1]
        
        return X,y
    



def normalize(matrix, dim=0):
    mean = matrix.mean(dim=0)
    std = matrix.std(dim=0)
    # 对每一列进行归一化
    normalized_tensor = (matrix - mean) / std
    return normalized_tensor

