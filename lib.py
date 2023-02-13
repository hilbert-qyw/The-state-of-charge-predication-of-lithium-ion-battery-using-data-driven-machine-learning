# coding = utf-8
"""
作者   : Hilbert
时间   :2021/11/12 15:31
"""
import sys
import os
from warnings import simplefilter

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]  # 上一级目录
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)
simplefilter(action='ignore', category=Warning)
simplefilter(action='ignore', category=FutureWarning)


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
import matplotlib.dates as mdates
import pickle
import scipy.stats as stats
import xlsxwriter
from openpyxl import load_workbook
from sklearn.metrics import mean_squared_error, r2_score
import random
from torch.utils import data
import torch
from torch import nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import copy


def mkdir(path):
    """
    mkdir of the path
    :param input: string of the path
    return: boolean
    """
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' is created!')
        return True
    else:
        print(path+' already exists!')
        return False


class DataAnalysis(object):
    """
    observe data distribution, sampling rate
    extract some important sensors data
    """
    def __init__(self, data, saving_path):
        self.data = data
        self.font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
        self.saving_path = saving_path
        # timestamp
        self.data['time'] = pd.to_datetime(self.data['time'])
        self.date = self.data['time'][0].strftime('%Y-%m-%d')
        self.saving_path = os.path.join(self.saving_path, self.date)
        mkdir(self.saving_path)

    def sensor_csv(self):
        self.data.to_csv(os.path.join(self.saving_path, fr'{self.date}_system_operation_data.csv'),
                         index=None, header=None, encoding='utf_8_sig')

    def sensor_info(self, saving=False):
        sensor_information = pd.DataFrame(columns=('index', 'information', 'total_number'))
        index = self.data['index']
        sensor_type = list(index.unique())
        for idx in sensor_type:
            sensor_index = index[index == idx].index.tolist()
            sensor_information = sensor_information.append(pd.DataFrame({'index': [idx],
                                                                         'information': [self.data.iloc[int(sensor_index[0])-1, 1]],
                                                                         'total_number': [len(sensor_index)]}),
                                                           ignore_index=True)
        if saving:
            sensor_information.to_csv(os.path.join(self.saving_path, fr'{self.date}_sensor_information.csv'),
                                      encoding='utf_8_sig', index=None)

        return sensor_information

    def sensor_data(self, number=None, plot=True) -> dict:
        # process data
        sensor_type = number if len(number) > 0 else list(self.data['index'].unique())
        print("总共" + str(len(sensor_type)) + "个数据")
        if plot:
            subfile = ['sensor_curve', 'sensor_hist', 'time_hist']
            filepath = []
            for file in subfile:
                filepath.append(os.path.join(self.saving_path, file))
                mkdir(filepath[-1])

        raw_sensor_data = {}
        for i, sensor_index in enumerate(sensor_type):
            print("=====>正在处理第" + str(i+1) + "个数据,请稍等")
            self.sensor_data = self.data.loc[self.data['index'][self.data['index'] == sensor_index].index]
            raw_sensor_data[self.sensor_data.iloc[0, 1]] = self.sensor_data[['time', 'current_data']]
            if len(self.sensor_data) == 0:
                continue

            if plot:
                self.sensor_curve(saving_path=filepath[0])
                self.sensor_hist(saving_path=filepath[1])
                delta_time = self.delta_time()
                self.time_hist(delta_time, saving_path=filepath[2])
            print("=====>第" + str(i + 1) + "个数据处理完毕！")
            print('\n')

        return raw_sensor_data

    def sensor_curve(self,saving_path):
        # sensor datas vary with time
        current_data = self.sensor_data['current_data']
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111)
        ax.plot(self.sensor_data['time'], current_data.values, lw=2)
        ax.set_title(self.sensor_data['sensor'][2], fontdict=self.font)
        ax.set_xlabel('time', fontdict=self.font)
        ax.set_ylabel(self.sensor_data['sensor'][2], fontdict=self.font)
        myFmt = mdates.DateFormatter('%H-%M-%S')
        ax.xaxis.set_major_formatter(myFmt)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.grid(alpha=0.4, linestyle=':')
        for tick in ax.get_xticklabels():
            tick.set_fontsize(10)
            tick.set_rotation(-20)
        plt.tight_layout()
        # plt.show()
        # plt.tight_layout()
        # plt.show()
        plt.savefig(saving_path + fr'\{self.date}_{self.sensor_data.iloc[0, 1]}随时间变化曲线.png')
        plt.close()

    def sensor_hist(self, saving_path):
        # the histgram of sensor data
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111)
        ax.hist(self.sensor_data['current_data'].values, bins=15)
        ax.set_title(self.sensor_data['sensor'][2], fontdict=self.font)
        ax.set_xlabel('Value', fontdict=self.font)
        ax.set_ylabel('Frequency', fontdict=self.font)
        # plt.tight_layout()
        # plt.show()
        plt.savefig(saving_path + fr'\{self.date}_{self.sensor_data.iloc[0, 1]}传感器数据分布直方图.png')
        plt.close()

    def delta_time(self):
        # calculate delta time
        delta_time = []
        time = pd.to_datetime(self.sensor_data['time'], format='%H-%M-%S')
        for idx in range(time.shape[0]-1):
            delta_time.append((time[idx+1] - time[idx]).total_seconds())
        return delta_time

    def time_hist(self, time, saving_path):
        # sampling rate
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111)
        ax.hist(time, bins=15)
        ax.set_title('Time', fontdict=self.font)
        ax.set_xlabel('Value', fontdict=self.font)
        ax.set_ylabel('Frequency', fontdict=self.font)
        # plt.tight_layout()
        # plt.show()
        plt.savefig(saving_path + fr'\{self.date}_{self.sensor_data.iloc[0, 1]}采样时间分布直方图.png')
        plt.close()


class DataReconstrction(object):
    def __init__(self, saving_path, raw_data: list, date: str, corr_excel=False):
        self.saving_path = saving_path
        self.raw_sensor = raw_data
        self.date = date
        self.corr_excel = corr_excel
        # if self.corr_excel:
        #     nan_excel = pd.DataFrame()
        #     nan_excel.to_excel(self.saving_path+'SOC_correlation_matrix.xlsx')
        #     self.writer = pd.ExcelWriter(self.saving_path + 'SOC_correlation_matrix.xlsx')


    def resample(self, interval=60, columns=None, saving_data=True) -> pd.DataFrame:
        """
        time series with different sampling rares --> interpolation --> resample(average)
        :param interval: sampling span
        :return: interpolated_sensor
        """
        interpolated_sensor = {}
        for sensor in self.raw_sensor:
            for key, value in sensor.items():
                ts = value.set_index('time')
                upsampled = ts.resample(f'{interval}s').mean()
                # interpolation + calculation
                interpolated = upsampled.interpolate(method='linear')
                interpolated_sensor[key] = interpolated

        aligned_interpolated_sensor = self.time_alignment(interpolated_sensor)

        if len(columns) == 0:
            columns = ['total_voltage', 'current', 'soc', 'avg_voltage', 'avg_temp', 'total_active', 'total_inactive',
                       'active', 'inactive']

        # convert to df
        keys = list(aligned_interpolated_sensor.keys())
        aligned_interpolated_sensor = pd.concat([aligned_interpolated_sensor[sensor] for sensor in keys], axis=1)
        aligned_interpolated_sensor.columns = columns

        if saving_data:
            with open(self.saving_path+fr'\{self.date}_SBMS_sensors.pk', 'wb') as f:
                pickle.dump(aligned_interpolated_sensor, f)

        return aligned_interpolated_sensor

    def time_alignment(self, day_data: dict) -> dict:
        """
        time series with different length --> extract start and end time in each variable --> Maximum sub-interval
        :param day_data:
        :return:
        """
        multi_variable = list(day_data.keys())
        for idx, variable in enumerate(multi_variable):
            if idx:
                temp_start, temp_end = day_data[variable].index[0], day_data[variable].index[-1]
                if (temp_start - start_time).total_seconds() > 0:
                    start_time = temp_start
                if (temp_end - end_time).total_seconds() < 0:
                    end_time = temp_end
            else:
                start_time, end_time = day_data[variable].index[0], day_data[variable].index[-1]

        self.aligned_data = {}
        for key, value in day_data.items():
            value = value.truncate(before=start_time, after=end_time)
            self.aligned_data[key] = value
        return self.aligned_data

    def correlation_coefficient(self, saving_corr=True):
        multi_variable = self.aligned_data.keys()
        sensors = pd.concat([self.aligned_data[sensor] for sensor in multi_variable], axis=1)
        sensors.columns = multi_variable
        corr_matrix = sensors.corr()
        SOC_corr = corr_matrix['SBMSSOC'].to_frame()
        SOC_corr.columns = [pd.to_datetime(self.date).strftime('%Y-%m-%d')]
        SOC_corr = SOC_corr.transpose()

        return SOC_corr


class Coulomb(object):
    def __init__(self, saving_path: str, sensors_data: dict, capacity, date: str):
        self.saving_path = saving_path
        self.sensors_data = sensors_data
        self.capacity = capacity
        self.date = date

    def coulomb_count(self):
        true_SOC = self.sensors_data['SBMSSOC'].values.squeeze()
        current = self.sensors_data['SBMS电流'].values.squeeze()
        coulomb_SOC = [true_SOC[0]]
        for idx in range(1, len(true_SOC)):
            coulomb_SOC.append(coulomb_SOC[-1] - 1 / 60 * current[idx] / self.capacity *100)
        self.plot_SOC(true_SOC, coulomb_SOC)

    def plot_SOC(self, true_SOC, predicted_SOC):
        rmse = np.sqrt(mean_squared_error(true_SOC, predicted_SOC))
        # print(rmse)
        figure1 = plt.figure(figsize=(16, 9))
        ax_1 = figure1.add_subplot(111)
        ax_1.plot(true_SOC, label='raw_SOC', lw=4)
        ax_1.plot(predicted_SOC, label='coulomb_SOC', lw=4)
        ax_1.legend()
        ax_1.set_title(fr'RMSE={rmse}')

        plt.tight_layout()
        plt.savefig(self.saving_path + fr'\{self.date}_SOC.png')
        # plt.show()

class DataGenerate(object):
    def __init__(self, days_data: dict, saving_path: str):
        self.days_data = days_data
        self.saving_path = saving_path

    def data_detection(self):
        """
        detect abnormal data in everyday
        if it doesn't meet the requirement, just delete it
        These data should concatenate together when their date are continuous.
        :return:
        """

        # abnormal detection
        normal_data = {}
        for key, value in self.days_data.items():
            today = value.index[0].strftime('%Y-%m-%d')
            ch_time = pd.to_datetime(today + ' 10:00:00')
            disch_time = pd.to_datetime(today + ' 18:00:00')

            # judge charging current value
            ch_value = value.truncate(before=ch_time, after=disch_time)
            ch_index = ch_value[ch_value['current'] < -10].index.tolist()

            # judge discharging current value
            dis_value = pd.concat((value.truncate(after=ch_time), value.truncate(before=disch_time)), axis=0)
            dis_index = dis_value[dis_value['current'] > 10].index.tolist()

            if len(ch_index) < 10 or len(dis_index) < 10:
                continue
            else:
                normal_data[key] = value
        return normal_data


    def data_segment(self, normal_data) -> list:
        # continuous days concatenate
        continuous_data = []
        keys = list(normal_data.keys())
        times = pd.to_datetime(keys)
        delta_time = [times[i+1] - times[i] for i in range(len(keys)-1)]
        days = [time.days for time in delta_time]
        index = [i+1 for i, val in enumerate(days) if val != 1]
        index.insert(0, 0)
        index.insert(len(keys),  len(keys))
        for idx in range(len(index)-1):
            continuous_day_data = pd.concat([normal_data[date] for date in keys[index[idx]:index[idx+1]]], axis=0)
            continuous_data.append(continuous_day_data)

        return continuous_data

    def slide_cycle(self, continuous_data, operation_mode='day'):
        cycle = {}
        ch_cycle = {}
        disch_cycle = {}
        for days_data in continuous_data:
            if operation_mode == 'day':
                days_data = days_data.to_period('D')
                times = days_data.resample('d').sum().index
                for time in times:
                    day_data = days_data[time:time]
                    time = time.strftime('%Y-%m-%d')
                    cycle[time] = day_data

            if operation_mode == 'hour':
                date = days_data.to_period('D')
                times = date.resample('d').sum().index
                for idx, time in enumerate(times):
                    nowday = time.strftime('%Y-%m-%d')
                    if idx < len(times) - 1:
                        tomorrow = times[idx+1].strftime('%Y-%m-%d')
                    else:
                        tomorrow = (times[idx] + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                    ch_start_time = pd.to_datetime(nowday + ' 10:00:00')
                    ch_end_time = pd.to_datetime(nowday + ' 18:00:00')
                    dis_end_time = pd.to_datetime(tomorrow + ' 10:00:00')
                    ch_cycle[nowday] = days_data[ch_start_time:ch_end_time]
                    if idx == 0:
                        start_time = pd.to_datetime(nowday + ' 0:00:00')
                        disch_cycle[nowday] = days_data[start_time:ch_start_time]
                        disch_cycle[tomorrow] = days_data[ch_end_time:dis_end_time]
                    else:
                        disch_cycle[tomorrow] = days_data[ch_end_time:dis_end_time]

        if operation_mode == 'day':
            with open(self.saving_path + '/day_cycle_data.pk', 'wb') as f:
                pickle.dump(cycle, f)
            return cycle

        if operation_mode == 'hour':
            with open(self.saving_path + '/state_cycle_data.pk', 'wb') as f:
                pickle.dump([ch_cycle, disch_cycle], f)
            return [ch_cycle, disch_cycle]


class Dataset(object):
    def __init__(self, raw_data:dict, saving_path, state, sequence_len, stride,padding=True,
                      padding_zero=False, saving_data=True):
        self.raw_data = raw_data
        self.saving_path = saving_path
        self.state = state
        self.keys = list(self.raw_data.keys())
        self.sequence_len = sequence_len
        self.stride = stride
        self.padding = padding
        self.padding_zero = padding_zero
        self.saving_data = saving_data


    def dataset_split(self, p_train=0.7, p_val=0.1):
        random.seed(5)
        idx = list(range(len(self.keys)))
        random.shuffle(idx)
        training_idx = idx[: int(len(idx) * p_train)]
        val_idx = idx[int(len(idx) * p_train):int(len(idx) * (p_train+p_val))]
        test_idx = idx[int(len(idx) * (p_train+p_val)):]

        # normalization value
        self.max, self.min = self.min_max_dataset(training_idx)

        # training
        self.saving_type = 'training'
        self.moving_window(training_idx)

        # validation
        self.saving_type = 'validation'
        self.moving_window(val_idx)

        # testing
        self.saving_type = 'testing'
        self.moving_window(test_idx)

    def min_max_dataset(self, indices):
        training_data = pd.concat([value for key, value in self.raw_data.items() if key in np.array(self.keys)[indices]])
        return training_data.max(), training_data.min()


    def moving_window(self, indices):
        """
        moving window to process time series
        :param input_data:
        :param input_path:
        :return:
        """
        sensor_data = {key: value for key, value in self.raw_data.items() if key in np.array(self.keys)[indices]}
        window_cycle = []

        # process each cycle data
        for key, value in sensor_data.items():
            value.reset_index(drop=True, inplace=True)
            if self.padding:
                if self.padding_zero:
                    zero = np.zeros((self.sequence_len-1, sensor_data.shape[1]))
                    value = pd.concat((pd.DataFrame(zero), value), axis=0)
                    value.reset_index(drop=True, inplace=True)
                else:
                    first_cycle = pd.DataFrame(np.tile(value.loc[0].values, (self.sequence_len-1, 1)))
                    first_cycle.columns = value.columns
                    value = pd.concat((first_cycle, value), axis=0)
                    value.reset_index(drop=True, inplace=True)

            # normalization
            value = (value - self.min) / (self.max - self.min)
            for j in range(0, value.shape[0] - self.sequence_len + 1, self.stride):
                features = value.drop(columns=['soc']).iloc[j:j + self.sequence_len, :].values
                targets = value['soc'].iloc[j + self.sequence_len-1]
                window_cycle.append([features, targets])

        if self.saving_data:
            with open(self.saving_path + fr'/{self.state}_{self.saving_type}_moving_window_{self.sequence_len}_'
                                         fr'{self.stride}.pk', 'wb') as f:
                pickle.dump(window_cycle, f)


def SOC_excel(saving_path, month_SOC_corr: list):
    # write to excel
    month_SOC_corr = pd.concat([SOC_corr for SOC_corr in month_SOC_corr], axis=0)
    writer = pd.ExcelWriter(saving_path + '\SOC_correlation_matrix.xlsx')
    month_SOC_corr.to_excel(excel_writer=writer, columns=month_SOC_corr.columns, header=True, encoding="utf-8", index=True)
    writer.save()
    writer.close()
    print("Successfully write to Excel!")

def month_data_curve(saving_path, month_sensors):
    # sensor data varies with month
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
    for key, value in month_sensors.items():
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111)
        ax.plot(value['time'], value['current_data'], lw=2)
        ax.set_title(key, fontdict=font)
        ax.set_xlabel('time', fontdict=font)
        ax.set_ylabel(key, fontdict=font)
        myFmt = mdates.DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_formatter(myFmt)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        ax.grid(alpha=0.4, linestyle=':')
        for tick in ax.get_xticklabels():
            tick.set_fontsize(10)
            tick.set_rotation(-20)
        plt.tight_layout()
        # plt.show()
        plt.savefig(saving_path + fr'\December\{key}随时间变化曲线.png')
        plt.close()


class ModelDataset(data.Dataset):
    def __init__(self, input_path, state, sequence_len, stride):
        self.input_path = input_path + fr'/day_{state}_moving_window_{sequence_len}_{stride}.pk'
        self.data = self.load_data()


    def load_data(self):
        # data's columns = 'total_voltage', 'current', 'avg_voltage', 'avg_temp', 'total_active', 'total_inactive',
        #                    'active', 'inactive'
        with open(self.input_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def __getitem__(self, index):
        features = self.data[index][0][:, [0, 1, 2, 3]].astype(np.float32)
        soc = np.array([self.data[index][1]]).astype(np.float32)
        # print(index)
        return torch.from_numpy(features), torch.from_numpy(soc)

    def __len__(self):
        return len(self.data)


class CNN_LSTM(nn.Module):
    def __init__(self,
                 sensor_num,
                 sequence_len,
                 lstm_hidden_size,
                 lstm_layers,
                 reg_hidden_size,
                 dropout=0.2,
                 encoder_bidirectional=False,
                 ):
        super(CNN_LSTM, self).__init__()
        # 1D CNN
        filter_1 = sensor_num * 2
        filter_2 = sensor_num * 4
        filter_3 = sensor_num * 8
        filter_4 = sensor_num * 16
        self.sensor_num = sensor_num
        self.sequence_len = sequence_len
        self.conv_layer1 = nn.Conv1d(in_channels=sensor_num, out_channels=filter_1, kernel_size=2, padding=1)
        self.max_pool_1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv_layer2 = nn.Conv1d(in_channels=filter_1, out_channels=filter_2, kernel_size=2, padding=1)
        self.max_pool_2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv_layer3 = nn.Conv1d(in_channels=filter_2, out_channels=filter_3, kernel_size=2, padding=1)
        self.max_pool_3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv_layer4 = nn.Conv1d(in_channels=filter_3, out_channels=filter_4, kernel_size=2, padding=1)
        self.max_pool_4 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # activation function
        self.relu = nn.ReLU()

        # LSTM
        self.lstm = nn.LSTM(input_size=sensor_num * 8,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=encoder_bidirectional)

        # regression
        self.reg_1 = nn.Linear(lstm_hidden_size, reg_hidden_size)
        self.reg_2 = nn.Linear(reg_hidden_size, 1)

    def forward(self, x):
        # print(x.shape,x.dtype)
        input = x.permute(0, 2, 1)
        # print(x.shape)
        conv1 = self.relu(self.conv_layer1(input))
        # print(conv1.shape)
        max_pool_1 = self.max_pool_1(conv1)
        # print(max_pool_1.shape)
        # conv2 = self.conv_layer2(max_pool_1)
        conv2 = self.relu(self.conv_layer2(max_pool_1))
        max_pool_2 = self.max_pool_2(conv2)
        # print(max_pool_2.shape)
        conv3 = self.conv_layer3(max_pool_2)
        # conv3 = self.relu(self.conv_layer3(max_pool_2))
        max_pool_3 = self.max_pool_3(conv3)
        max_pool_3 = max_pool_3.permute(0, 2, 1)
        # # print(max_pool_3.shape)
        # conv4 = self.conv_layer4(max_pool_3)
        # max_pool_4 = self.max_pool_4(conv4)
        # print(x.shape)
        lstm_out, _ = self.lstm(max_pool_3)
        # print(x.shape)
        reg_input = lstm_out[:, -1, :]
        reg_1_output = self.reg_1(reg_input)
        output = self.reg_2(reg_1_output)
        return output


class Trainer(object):
    def __init__(
            self,
            model: nn.Module,
            criterion=None,
            train_dataloader=None,
            *,
            scheduler=None,
            device='cpu',
            verbose=True,
            saving_path='./results',
            val_dataloader=None,
            test_dataloader=None,

    ) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.device = device
        self.scheduler = scheduler
        self.verbose = verbose
        self.saving_path = saving_path

        if val_dataloader is not None:
            self.val_dataloader = val_dataloader

        if test_dataloader is not None:
            self.test_dataloader = test_dataloader

    def train(self, epochs=50, optimizer=None, ):
        if optimizer is not None:
            self.optimizer = optimizer

        print("=> Beginning training")

        train_loss = []
        val_loss = []
        best_loss = None
        test_rmse = None

        self.model.train()

        for epoch in range(epochs):
            train_batch_loss = []
            print('========Epoch(train)-%d========' % epoch)
            # for input, tar in tqdm(self.train_dataloader, desc='Epoch(train)-%d' % epoch):
            for input, tar in self.train_dataloader:
                self.optimizer.zero_grad()
                pre = self.model(input)
                loss = self.criterion(pre, tar)

                loss.backward()
                self.optimizer.step()

                train_batch_loss.append(loss.item() / tar.shape[0])

            if self.scheduler is not None:
                self.scheduler.step()

            train_epoch_loss = sum(train_batch_loss) / len(train_batch_loss)

            val_epoch_loss, val_err, _ = self.eval(self.val_dataloader)

            if self.verbose:
                print(f'Epoch:{epoch:3d}\nTraining Loss:{train_epoch_loss:.4f}\tValidation Loss:{val_epoch_loss:.4f}',
                      flush=True)
                print(f'Validation metrics:\nRMSE:{val_err[0]:.4f}\tMAE:{val_err[1]:.4f}'
                      f'\tR2:{val_err[2]:.4f}', flush=True)

            if best_loss == None or val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                self.best_model = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                print("successfully save best model!")

            train_loss.append(train_epoch_loss)
            val_loss.append(val_epoch_loss)

            self.last_model = self.model.state_dict()
            tmp_rmse = self.test()
            if test_rmse == None or tmp_rmse < test_rmse:
                test_rmse = tmp_rmse
                test_rmse_epoch = epoch

        self.history = {'train loss': train_loss, 'val loss': val_loss}

        print("==> Best test RMSE is:", test_rmse, "\nepoch=", test_rmse_epoch)
        print("=> Saving model to file")

        if not os.path.exists(self.saving_path):
            os.mkdir(self.saving_path)
        torch.save(self.model.state_dict(), os.path.join(self.saving_path, 'last_model.pt'))
        torch.save(self.best_model, os.path.join(self.saving_path, f'best_model_{self.best_epoch}.pt'))
        torch.save(self.history, os.path.join(self.saving_path, 'loss_history.pt'))

        self.plot_loss()

        return self.history

    def eval(self, data_loader, save_data=False, save_plot=False, name=None):
        self.model.eval()
        with torch.no_grad():
            y_true = []
            y_predict = []
            cum_loss = []
            for input, tar in data_loader:
                pre = self.model(input)
                loss = self.criterion(pre, tar)

                cum_loss.append(loss.item() / tar.shape[0])

                y_true.append(tar.detach().numpy())
                y_predict.append(pre.detach().numpy())

            val_epoch_loss = sum(cum_loss) / len(cum_loss)

        y_true = np.concatenate(y_true)
        # print(y.shape)
        y_predict = np.concatenate(y_predict)

        mae = mean_absolute_error(y_true, y_predict)
        mse = mean_squared_error(y_true, y_predict)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_predict)

        # print(len(y))

        if save_data:
            np.savetxt(os.path.join(self.saving_path, name + '_label.txt'), y_true)
            np.savetxt(os.path.join(self.saving_path, name + '_predict.txt'), y_predict)
            with open(os.path.join(self.saving_path, name + '_metrics.txt'), 'w') as f:
                print(f'\tRMSE:{rmse}, MAE:{mae}, R2:{r2}', file=f)

        if save_plot:
            plt.figure()
            plt.plot(y_true, 'k', label='target')
            plt.plot(y_predict, 'r', label='predict')
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.saving_path + '/' + name + '_target_predict.png')
            plt.cla()

            plt.plot(y_true - y_predict)
            plt.ylabel('prediction error')
            plt.tight_layout()
            plt.savefig(self.saving_path + '/' + name + '_target_predict_error.png')
            plt.clf()

            plt.scatter(y_true, y_predict)
            plt.xlabel('target value')
            plt.ylabel('predicted value')
            plt.tight_layout()
            plt.savefig(self.saving_path + '/' + name + '_target_predict_scatter.png')

            plt.clf()

        return val_epoch_loss, (rmse, mae, r2), (y_true, y_predict)

    def test(self, mode='last', save_data=False, save_plot=False):
        print("\n=> Evaluating " + mode + " model on test dataset")

        if mode == 'last':
            model = self.last_model
        else:
            model = self.best_model

        self.model.load_state_dict(model)
        test_loss, metrics, y = self.eval(self.test_dataloader, save_data=save_data, save_plot=save_plot, name=mode)

        y_true = y[0]
        y_pred = y[1]
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        print(f'Test loss:{test_loss}, RMSE:{rmse}, MAE:{mae}, R2:{r2}')
        return rmse


    def plot_loss(self):
        epoch_arr = list(range(len(self.history['train loss'])))

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(epoch_arr, self.history['train loss'], )
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('train loss')

        plt.subplot(2, 1, 2)
        plt.plot(epoch_arr, self.history['val loss'], )
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('val loss')

        savepath = os.path.join(self.saving_path, 'train_loss.png')
        plt.savefig(savepath)
        plt.clf()



