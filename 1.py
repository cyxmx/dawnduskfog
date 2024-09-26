import numpy as np
import math
import xarray as xr
from numpy import deg2rad, rad2deg, arctan, arcsin, tan, sqrt, cos, sin
import datetime
import matplotlib.pyplot as plt

# 常量
ea = 6378.137  # 地球的半长轴[km]
eb = 6356.7523  # 地球的短半轴[km]
h = 42164  # 地心到卫星质心的距离[km]
λD = deg2rad(104.7)  # 卫星星下点所在经度
# 列偏移
COFF = {"0500M": 10991.5,
        "1000M": 5495.5,
        "2000M": 2747.5,
        "4000M": 1373.5}
# 列比例因子
CFAC = {"0500M": 81865099,
        "1000M": 40932549,
        "2000M": 20466274,
        "4000M": 10233137}
LOFF = COFF  # 行偏移
LFAC = CFAC  # 行比例因子

# 地理经纬度116.2°E~129.0°E，28.7°N~41.5°N
lat_real = [28.7, 41.5]
lon_real = [116.2, 129.0]

def day_of_year(date):
    """计算给定日期是一年中的第几天"""
    return date.timetuple().tm_yday

def solar_declination(day_of_year):
    """根据一年中的第几天计算太阳赤纬"""
    return 23.44 * np.sin(np.deg2rad((360 / 365.0) * (day_of_year - 81)))

def hour_angle(longitude, time):
    """根据经度和当地时间计算太阳时角"""
    # 计算太阳时间（solar time），基于经度调整的当地时间
    solar_time = time.hour + time.minute / 60 + time.second / 3600 + longitude / 15.0
    return np.deg2rad(15 * (solar_time - 12))

def parse_time_string(time_string):
    """将 hhmmss 格式的字符串转换为时间对象"""
    if len(time_string) != 6:
        raise ValueError("时间字符串格式应为 hhmmss,长度应为6位。")
    hour = int(time_string[:2])
    minute = int(time_string[2:4])
    second = int(time_string[4:6])
    return datetime.time(hour=hour, minute=minute, second=second)

def solar_zenith_angle(lat_matrix, lon_matrix, date_time_str):
    """根据二维纬度和经度矩阵计算每个点的太阳天顶角"""
    # 当前日期
    today = datetime.date.today()
    
    # 将时间字符串解析为 time 对象
    time_obj = parse_time_string(date_time_str)
    
    # 将日期和时间组合为 datetime 对象
    date_time = datetime.datetime.combine(today, time_obj)
    
    # 获取一年中的第几天
    day = day_of_year(date_time)

    # 计算太阳赤纬
    delta = solar_declination(day)

    # 计算太阳时角
    H_matrix = hour_angle(lon_matrix, date_time)

    # 将纬度转换为弧度
    lat_rad_matrix = np.deg2rad(lat_matrix)

    # 太阳天顶角公式（矩阵运算）
    cos_theta_matrix = (np.sin(lat_rad_matrix) * np.sin(np.deg2rad(delta)) +
                        np.cos(lat_rad_matrix) * np.cos(np.deg2rad(delta)) * np.cos(H_matrix))
    
    # 计算天顶角并返回结果矩阵
    return np.rad2deg(np.arccos(cos_theta_matrix))

def getsoz(filename, geo_desc):
    lat_S, lat_N, lon_W, lon_E, step = [1000 * x for x in geo_desc]
    lat = np.arange(lat_N, lat_S, -step) / 1000 # lat.shape = (256,)
    lon = np.arange(lon_W, lon_E, step) / 1000 # lon.shape = (256,)
    lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    # date_time = datetime.datetime.now()  # 当前时间
    # date_time_end = filename[-22:-16] # 结束测量时间
    date_time_start = filename[-37:-31] # 开始测量时间
    soz_matrix = solar_zenith_angle(lat_mesh, lon_mesh, date_time_start)
    # soz_matrix = solar_zenith_angle(lat_mesh, lon_mesh, date_time_end)
    return soz_matrix

def latlon2linecolumn(lat, lon, resolution):
        """
        (lat, lon) → (line, column)
        resolution：文件名中的分辨率{'0500M', '1000M', '2000M', '4000M'}
        line, column不是整数
        """
        # Step1.检查地理经纬度
        # Step2.将地理经纬度的角度表示转化为弧度表示
        lat = deg2rad(lat)
        lon = deg2rad(lon)
        # Step3.将地理经纬度转化成地心经纬度
        eb2_ea2 = eb**2 / ea**2
        λe = lon
        φe = arctan(eb2_ea2 * tan(lat))
        # Step4.求Re
        cosφe = cos(φe)
        re = eb / sqrt(1 - (1 - eb2_ea2) * cosφe**2)
        # Step5.求r1,r2,r3
        λe_λD = λe - λD
        r1 = h - re * cosφe * cos(λe_λD)
        r2 = -re * cosφe * sin(λe_λD)
        r3 = re * sin(φe)
        # Step6.求rn,x,y
        rn = sqrt(r1**2 + r2**2 + r3**2)
        x = rad2deg(arctan(-r2 / r1))
        y = rad2deg(arcsin(-r3 / rn))
        # Step7.求c,l
        column = COFF[resolution] + x * 2**-16 * CFAC[resolution]
        line = LOFF[resolution] + y * 2**-16 * LFAC[resolution]
        return line, column

def merged_channel(file):
    channel01 = file.extract('Channel01', calibration='reflectance') # 0.47μm
    channel02 = file.extract('Channel02', calibration='reflectance') # 0.65μm
    channel03 = file.extract('Channel03', calibration='reflectance') # 0.825μm
    channel04 = file.extract('Channel04', calibration='reflectance') # 1.379μm
    channel05 = file.extract('Channel05', calibration='reflectance') # 1.61μm
    channel06 = file.extract('Channel06', calibration='reflectance') # 2.25μm
    channel07 = file.extract('Channel07', calibration='radiance') # 3.75μm(high)
    channel08 = file.extract('Channel07', calibration='radiance') # 3.75μm(low)
    channel09 = file.extract('Channel09', calibration='radiance') # 6.25μm
    channel10 = file.extract('Channel10', calibration='radiance') # 6.95μm
    channel11 = file.extract('Channel11', calibration='radiance') # 7.42μm
    channel12 = file.extract('Channel12', calibration='radiance') # 8.55μm
    channel13 = file.extract('Channel13', calibration='radiance') # 10.80μm
    channel14 = file.extract('Channel14', calibration='radiance') # 12.00μm
    channel15 = file.extract('Channel15', calibration='radiance') # 13.3μm
    #将所有channel进行合并
    merged_channel = channel01
    for i in range(2, 16):
        channel = locals()[f'channel{i:02d}']
        merged_channel = xr.concat([merged_channel, channel], dim="channels")
    # merged_channel = xr.concat([channel01, channel02, channel03, channel04, channel05, channel06, channel07, channel08, channel09, channel10, channel11, channel12, channel13, channel14, channel15], dim="dim")
    # channels = [f"channel{i:02d}" for i in range(1, 16) if i != 7]
    channels = [f"channel{i:02d}" for i in range(1, 16)]
    merged_channel = merged_channel.assign_coords(channels = ("channels", channels)) # 更改coords
    merged_channel = merged_channel.rename('Merged Channels') # 重新命名
    return merged_channel

def getfog(merge_channel, SOZ):
    B1 = merge_channel.sel(channels='channel01') # h8:0.47μm, fy4b:0.47μm
    B3 = merge_channel.sel(channels='channel03') # h8:0.64μm, fy4b:0.825μm(channel2:0.65μm)
    B4 = merge_channel.sel(channels='channel04') # h8:0.86μm, fy4b:1.379μm(channel3:0.825μm)
    B5 = merge_channel.sel(channels='channel05') # h8:1.6μm, fy4b:1.61μm
    B7 = merge_channel.sel(channels='channel07') # h8:3.9μm, fy4b:3.75μm(high)
    B14 = merge_channel.sel(channels='channel14') # h8:11.2μm, fy4b:12.00μm(channel13:10.80μm)
    
    R_Sum = B1 + B4 # type(R_Sum)=xarray.DataArray
    R_Dvalue = np.abs(B4 - B1)
    DE = B3 - B5
    BTD = B7 - B14
    Th1 = SOZ*-0.694+338.296-3
    Th2 = SOZ*-0.0179+1.615-0.015
    Th3 = 0.015
    Th4 = 0.004
    Th5 = SOZ*-0.633+52.451-3
    Th6 = SOZ*-0.633+52.451+3

    result = np.zeros((256,256), dtype=int)
    mask1 = B7 > Th1
    mask2 = (R_Dvalue < Th3) | (R_Sum > Th2)
    mask3 = (DE < 0.004) & (Th5 < BTD) & (BTD < Th6)
    final_mask = mask1 & mask2 & mask3
    result[final_mask] = 1
    return(result)

def getfog_fillvalue(merge_channel, SOZ, FillValue):
    B1 = merge_channel.sel(channels='channel01') # h8:0.47μm, fy4b:0.47μm
    B3 = merge_channel.sel(channels='channel03') # h8:0.64μm, fy4b:0.825μm(channel2:0.65μm)
    B4 = merge_channel.sel(channels='channel04') # h8:0.86μm, fy4b:1.379μm(channel3:0.825μm)
    B5 = merge_channel.sel(channels='channel05') # h8:1.6μm, fy4b:1.61μm
    B7 = merge_channel.sel(channels='channel07') # h8:3.9μm, fy4b:3.75μm(high)
    B14 = merge_channel.sel(channels='channel14') # h8:11.2μm, fy4b:12.00μm(channel13:10.80μm)
    
    SOZ = np.where((SOZ >= 81) & (SOZ < 90), SOZ, FillValue)
    
    R_Sum = B1 + B4 # type(R_Sum)=xarray.DataArray
    R_Dvalue = np.abs(B4 - B1)
    DE = B3 - B5
    BTD = B7 - B14
    Th1 = np.where(SOZ == FillValue, FillValue, SOZ*-0.694+338.296-3)
    Th2 = np.where(SOZ == FillValue, FillValue, SOZ*-0.0179+1.615-0.015)
    Th3 = 0.015
    Th4 = 0.004
    Th5 = np.where(SOZ == FillValue, FillValue, SOZ*-0.633+52.451-3)
    Th6 = np.where(SOZ == FillValue, FillValue, SOZ*-0.633+52.451+3)

    final_mask = np.zeros((256,256), dtype=int)
    # mask1 = B7 > Th1
    mask1 = np.where((B7 == FillValue) | (Th1 == FillValue), FillValue, np.where(B7 > Th1, 1, 0))
    # mask2 = (R_Dvalue < Th3) | (R_Sum > Th2)
    mask2a = np.where((R_Dvalue == FillValue) | (Th3 == FillValue), FillValue, np.where(R_Dvalue < Th3, 1, 0))
    mask2b = np.where((R_Sum == FillValue) | (Th2 == FillValue), FillValue, np.where(R_Sum > Th2, 1, 0))
    mask2 = np.where((mask2a == FillValue) | (mask2b == FillValue), FillValue, mask2a | mask2b)
    # mask3 = (DE < 0.004) & (Th5 < BTD) & (BTD < Th6)
    mask3a = np.where(DE == FillValue, FillValue, np.where(DE < 0.004, 1, 0))
    mask3b = np.where((Th5 == FillValue) | (BTD == FillValue), FillValue, np.where(Th5 < BTD, 1, 0))
    mask3c = np.where((Th6 == FillValue) | (BTD == FillValue), FillValue, np.where(BTD < Th6, 1, 0))
    mask3 = np.where((mask3a == FillValue) | (mask3b == FillValue) | (mask3c == FillValue), FillValue, mask3a & mask3b & mask3c)
    # final_mask = mask1 & mask2 & mask3
    final_mask = np.where((mask1 == FillValue) | (mask2 == FillValue) | (mask3 == FillValue), FillValue, mask1 & mask2 & mask3)
    
    return(final_mask)

def show_figure(data):
    # 定义颜色映射
    # 0 -> 白色, 1 -> 蓝色, 65535 -> 黑色（或其他颜色）
    cmap = plt.cm.colors.ListedColormap(['white', 'blue', 'black'])
    bounds = [0, 1, 2, 3]  # 界限
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    # 创建可视化
    plt.figure(figsize=(6, 6))
    plt.imshow(data, cmap=cmap, norm=norm)
    plt.colorbar(ticks=[0, 1, 2], label='FogCondition')
    plt.title('DawnDuskFog')
    plt.xlabel('lon')
    plt.ylabel('lat')
    plt.xticks([])  # 隐藏 x 轴刻度
    plt.yticks([])  # 隐藏 y 轴刻度
    plt.show()

class FY4B_AGRI_L1(object):
    def __init__(self, file_path, geo_desc=None):
        # 读取HDF文件，通过传递 group 参数来选择特定的数据集
        self.dataset = xr.open_dataset(file_path, group = '/')
        self.ds = xr.open_dataset(file_path, group = '/Data')
        self.cal = xr.open_dataset(file_path, group = '/Calibration')
        self.resolution = file_path[-15:-10] # resolution = 4000M
        self.line_begin = self.dataset.attrs['Begin Line Number'] # HDF全局文件属性：起始行号Begin Line Number
        self.line_end = self.dataset.attrs['End Line Number'] # HDF全局文件属性：结束行号End Line Number
        self.column_begin = self.dataset.attrs['Begin Pixel Number'] # HDF全局文件属性：起始象元号Begin Pixel Number
        self.column_end = self.dataset.attrs['End Pixel Number'] # HDF全局文件属性：结束象元号End Pixel Number
        self.set_geo_desc(geo_desc)
        '''
        ==========可以使用如下方式查看全局文件属性===========
        file_path = f'./data-AGRI/FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20240915193000_20240915194459_4000M_V0001.HDF'
        with h5py.File(file_path, 'r') as hdf:
            # 打印所有全局文件属性
            print("Global Attributes:")
            for attr in hdf.attrs:
                print(f"{attr}: {hdf.attrs[attr]}")
        '''

    def __del__(self):
        # 关闭文件
        self.dataset.close()
        self.ds.close()
        self.cal.close()

    def set_geo_desc(self, geo_desc):
        if geo_desc is None:
            self.line = self.column = self.geo_desc = None
            return
        # 先乘1000取整是为了减少浮点数的精度误差累积问题
        lat_S, lat_N, lon_W, lon_E, step = [1000 * x for x in geo_desc]
        # lat = np.arange(lat_N, lat_S - 1, -step) / 1000
        # lon = np.arange(lon_W, lon_E + 1, step) / 1000
        lat = np.arange(lat_N, lat_S, -step) / 1000 # lat.shape = (256,)
        lon = np.arange(lon_W, lon_E, step) / 1000 # lon.shape = (256,)
        lon_mesh, lat_mesh = np.meshgrid(lon, lat)
        # 求geo_desc对应的标称全圆盘行列号
        line, column = latlon2linecolumn(lat_mesh, lon_mesh, self.resolution) # resolution = 4000M
        self.line = xr.DataArray(line, coords=(('lat', lat), ('lon', lon)), name='line')
        '''
        line = <xarray.DataArray(lat:256, lon:256)>
        array([line])
        Coorinates:
            * lat [...]
            * lon [...]
        '''
        self.column = xr.DataArray(column, coords=(('lat', lat), ('lon', lon)), name='column')
        '''
        column = <xarray.DataArray(lat:256, lon:256)>
        array([column])
        Coorinates:
            * lat [...]
            * lon [...]
        '''
        self.geo_desc = geo_desc

    def extract(self, channel_name, calibration='reflectance',
                geo_desc=None, interp_method='nearest'):
        """
        按通道名和定标方式提取geo_desc对应的数据
        channel_name：要提取的通道名（如'Channel01'）

        calibration: {'dn', 'reflectance', 'radiance', 'brightness_temperature'}
        """
        if geo_desc and geo_desc != self.geo_desc:
            self.set_geo_desc(geo_desc) # 如果没有转换行列号，则进行转换
        dn = self.ds[f'NOM{channel_name}'][:] # ds是xr.DataArray的group='/Data'，提取NOMChannel01~NOMChannel15：xxxμm通道4KM图像数据层
        line = [i for i in range(self.line_begin, self.line_end + 1)] # 提取有效行号
        column = [i for i in range(self.column_begin, self.column_end + 1)] # 提取有效象元号
        dn_values = xr.DataArray(dn, coords=[line, column], dims=['line', 'column'])
        '''
        dn_values = <xarray.DataArray(line:有效行号维度, column:有效象元号维度)>
        array([dn])
        Coorinates:
            * line [有效行号]
            * column [有效象元号]
        '''
        if self.geo_desc:
            # 若geo_desc已指定，则插值到对应网格
            dn_values = dn_values.interp(line=self.line
                                         , column=self.column
                                         , method=interp_method) # 按照上述line和column的有效行号和象元号插值（截取）出有效的dn_value
            del dn_values.coords['line'], dn_values.coords['column'] # 替换完dn_value后 把有效行号和象元号删除
        else:
            # 若geo_desc为None，则保留原始NOM网格
            pass
        return dn_values # 直接输出原始值
        # return self.calibrate(channel_name, calibration, dn_values) # 输出由原始值计算转化后的反射率或者辐亮度
    
    def calibrate(self, channel_name, calibration, dn_values):
        """
        前面6个通道，用查找表和系数算出来都是反射率reflectance，后面用查找表是亮温brightness_temperature，用系数是辐亮度radiance。
        """
        if calibration == 'dn':
            dn_values.attrs = {'units': 'DN'}
            return dn_values
        channel_num = int(channel_name[-2:])
        dn_values = dn_values.fillna(dn_values.FillValue)  # 保留缺省值,.fillna() 用于在数据集中使用FillValue替换缺失值（NaN）
        '''
        dn_values是xxxμm通道4KM图像数据层
        其中具有的Attr如下,具体参考文件中各科学数据集中SDS Attribute
        valid_range:(0,4095)
        FillValue:65535
        Intercept:0.0
        Slope:1.0
        center_wavelength:xxxμm
        '''
        if ((calibration == 'reflectance' and channel_num <= 6) or
                (calibration == 'radiance' and channel_num > 6)): # 反射通道（1~6）,用于将DN转换成反射率;热红外通道,用于将DN转换成辐亮度。
            k, b = self.cal['CALIBRATION_COEF(SCALE+OFFSET)'].values[channel_num - 1] # self.cal是Calibration数据集,cal['CALIBRATION_COEF(SCALE+OFFSET)']指SDS16，意思是找当前通道的斜率k截距b
            data = k * dn_values.where(dn_values != dn_values.FillValue) + b # 将FillValue去掉后通过kx+b计算出data（反射率或者是辐亮度）
            data.attrs['units'] = '100%' if calibration == 'reflectance' else 'mW/ (m2 cm-1 sr)' # 反射率以0%到100%表示，mW/(m²·cm⁻¹·sr) 用于描述光谱辐射度或辐射亮度的强度
        elif calibration == 'brightness_temperature' and channel_num > 6: # 亮温查找表，并且通过插值（method=linear）算出有效值，单位为开尔文K
            cal_table = self.cal[f'CAL{channel_name}']
            cal_table = cal_table.swap_dims({cal_table.dims[0]: 'dn'})
            data = cal_table.interp(dn=dn_values)
            del data.coords['dn']
            data.attrs = {'units': 'K'}
        else:
            raise ValueError(f'{channel_name}没有{calibration}的定标方式')
        data.name = f'{channel_name}_{calibration}'
        return data # 输出由原始值计算转化后的反射率或者辐亮度


if __name__ == "__main__":
    # filename = f'./data-AGRI/FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20240915091500_20240915092959_4000M_V0001.HDF'
    filename = f'./data-AGRI/FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20240915093000_20240915094459_4000M_V0001.HDF'
    # filename = f'./data-AGRI/FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20240915094500_20240915095959_4000M_V0001.HDF'
    # filename = f'./data-AGRI/FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20240915100000_20240915101459_4000M_V0001.HDF'
    # filename = f'./data-AGRI/FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20240915101500_20240915102959_4000M_V0001.HDF'
    
    geo_desc = [28.7, 41.5, 116.2, 129.0, 0.05] # lat_S, lat_N, lon_W, lon_E, step
    file = FY4B_AGRI_L1(filename, geo_desc)
    
    """
    =====第一题 读取黄渤海经纬度数据=====
    """
    merge_channel = merged_channel(file) # type(merge_channel)=xarray.DataArray
    print(merge_channel) # merge_channel.shape = (channels: 15, lat: 256, lon: 256)

    """
    =====第二题 获取SOZ太阳高度角的值=====
    """
    SOZ = getsoz(filename, geo_desc) # type(SOZ) = numpy.ndarray
    # print(SOZ) # SOZ.shape = (256, 256)

    """
    =====第三题 动态阈值识别=====
    因为通道波段没对齐，代码逻辑通但识别仍然有问题
    """
    FillValue = 65535

    # dawnduskfog = getfog(merge_channel, SOZ) # type(dawnduskfog) = numpy.ndarray
    # print(dawnduskfog) # dawnduskfog.shape = (256, 256)
    
    dawnduskfog = getfog_fillvalue(merge_channel, SOZ, FillValue)
    # print(dawnduskfog)
    show_figure(dawnduskfog)
    