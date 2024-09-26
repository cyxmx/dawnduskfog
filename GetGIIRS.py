import numpy as np
import math
import xarray as xr
from numpy import deg2rad, rad2deg, arctan, arcsin, tan, sqrt, cos, sin
from scipy.interpolate import griddata

def GetSolarZenithVIS(hdf_file):
    print('===== Solar_Zenith_VIS_Group =====')
    Solar_Zenith_VIS = hdf_file['Geolocation']['Solar_Zenith_VIS']
    # print(Solar_Zenith_VIS.shape) # 512 * 512
    data = Solar_Zenith_VIS[:]
    Altitude_VIS = np.full(data.shape, 90) - data # 太阳高度角和太阳天顶角互余90°
    # print(Altitude_VIS)
    return Altitude_VIS # 输出为太阳高度角

def GetLatitudeVIS(hdf_file):
    # print('===== Latitude_VIS_Group =====')
    Latitude_VIS = hdf_file['Geolocation']['Latitude_VIS']
    # print(Latitude_VIS.shape) # 512 * 512
    data = Latitude_VIS[:]
    # print(Latitude_VIS[:])
    return data

def GetLongitudeVIS(hdf_file):
    # print('===== Longitude_VIS_Group =====')
    Longitude_VIS = hdf_file['Geolocation']['Longitude_VIS']
    # print(Longitude_VISS.shape) # 512 * 512
    data = Longitude_VIS[:]
    # print(Longitude_VIS[:])
    return data

def GetGeolocation(hdf_file):
    print('===== Geolocation_Group =====')
    # 读取Geolocation Group
    Geolocation = hdf_file['Geolocation']
    for key in Geolocation.keys():
        print(Geolocation[key])


class FY4B_GIIRS_L1(object):
    def __init__(self, file_path, geo_desc=None):
        # 读取HDF文件，通过传递 group 参数来选择特定的数据集
        self.dataset = xr.open_dataset(file_path, group = '/')
        self.geo = xr.open_dataset(file_path, group = '/Geolocation')
        self.data = xr.open_dataset(file_path, group = '/Data')
        self.qa = xr.open_dataset(file_path, group = '/QA')
        self.resolution = '12000M' # resolution = 12KM
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
        self.dataset.close()
        self.geo.close()
        self.data.close()
        self.qa.close()

    def set_geo_desc(self, geo_desc):
        if geo_desc is None:
            self.line = self.column = self.geo_desc = None
            print("geo_desc is None")
            return
        # 先乘1000取整是为了减少浮点数的精度误差累积问题
        lat_S, lat_N, lon_W, lon_E, step = [1000 * x for x in geo_desc]
        # lat = np.arange(lat_N, lat_S - 1, -step) / 1000
        # lon = np.arange(lon_W, lon_E + 1, step) / 1000
        lat = np.arange(lat_N, lat_S, -step) / 1000 # lat.shape = (256,)
        lon = np.arange(lon_W, lon_E, step) / 1000 # lon.shape = (256,)
        lon_mesh, lat_mesh = np.meshgrid(lon, lat)
        # 求geo_desc对应的标称全圆盘行列号
        # line, column = latlon2linecolumn(lat_mesh, lon_mesh, self.resolution) # resolution = 4000M
        self.line = xr.DataArray(lon_mesh, coords=(('lat', lat), ('lon', lon)), name='lon_mesh')
        '''
        line = <xarray.DataArray(lat:256, lon:256)>
        array([line])
        Coorinates:
            * lat [...]
            * lon [...]
        '''
        self.column = xr.DataArray(lat_mesh, coords=(('lat', lat), ('lon', lon)), name='lat_mesh')
        '''
        column = <xarray.DataArray(lat:256, lon:256)>
        array([column])
        Coorinates:
            * lat [...]
            * lon [...]
        '''
        self.geo_desc = geo_desc
        # print("set geodesc")

    def extract(self, geo_desc=None, interp_method='linear'):
        """
        按通道名和定标方式提取geo_desc对应的数据
        channel_name：要提取的通道名（如'Channel01'）

        calibration: {'dn', 'reflectance', 'radiance', 'brightness_temperature'}
        """
        if geo_desc and geo_desc != self.geo_desc:
            self.set_geo_desc(geo_desc)
            # print("set geodesc")
        Latitude_VIS = self.geo['Latitude_VIS'][:]
        Longitude_VIS = self.geo['Longitude_VIS'][:]
        Solar_Zenith_VIS = self.geo['Solar_Zenith_VIS'][:]
        line = [i for i in range(self.line_begin, self.line_end + 1)] # 提取有效行号
        column = [i for i in range(self.column_begin, self.column_end + 1)] # 提取有效象元号
        Latitude_VIS_values = xr.DataArray(Latitude_VIS, dims=['line', 'column'])
        Longitude_VIS_values = xr.DataArray(Longitude_VIS, dims=['line', 'column'])
        Solar_Zenith_VIS_values = xr.DataArray(Solar_Zenith_VIS, dims=['line', 'column'])
        '''
        dn_values = <xarray.DataArray(line:有效行号维度, column:有效象元号维度)>
        array([dn])
        Coorinates:
            * line [有效行号]
            * column [有效象元号]
        '''
        if self.geo_desc:
            # print("set geodesc")
            # 若geo_desc已指定，则插值到对应网格
            Latitude_VIS_values = Latitude_VIS_values.interp(line=self.line
                                         , column=self.column
                                         , method=interp_method) # 按照上述line和column的有效行号和象元号插值（截取）出有效的dn_value
            del Latitude_VIS_values.coords['line'], Latitude_VIS_values.coords['column'] # 替换完dn_value后 把有效行号和象元号删除
            Longitude_VIS_values = Longitude_VIS_values.interp(line=self.line
                                         , column=self.column
                                         , method=interp_method) # 按照上述line和column的有效行号和象元号插值（截取）出有效的dn_value
            del Longitude_VIS_values.coords['line'], Longitude_VIS_values.coords['column'] # 替换完dn_value后 把有效行号和象元号删除
            Solar_Zenith_VIS_values = Solar_Zenith_VIS_values.interp(line=self.line
                                         , column=self.column
                                         , method=interp_method) # 按照上述line和column的有效行号和象元号插值（截取）出有效的dn_value
            del Solar_Zenith_VIS_values.coords['line'], Solar_Zenith_VIS_values.coords['column'] # 替换完dn_value后 把有效行号和象元号删除
            print(Solar_Zenith_VIS_values)
            # print(Latitude_VIS_values)
            # points = np.vstack((Latitude_VIS_values.values.ravel(), Longitude_VIS_values.values.ravel())).T
            # values = Solar_Zenith_VIS_values.values.ravel()
            # new_points = np.vstack((self.line.values.ravel(), self.column.values.ravel())).T
            # Solar_Zenith_VIS_new_values = griddata(points, values, new_points, method=interp_method)
            # Solar_Zenith_VIS_new_values = Solar_Zenith_VIS_new_values.reshape(self.line.shape)
            # Solar_Zenith_VIS_new = xr.DataArray(Solar_Zenith_VIS_new_values, dims=['line', 'column'])
            # print("transfer conplete")

        else:
            # 若geo_desc为None，则保留原始NOM网格
            pass
        return Solar_Zenith_VIS_values  # 直接输出插值xarray和经纬度xarray
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
    filename = f'./data-GIIRS/FY4B-_GIIRS-_N_REGC_1050E_L1-_IRD-_MULT_NUL_20240915192437_20240915192945_012KM_021V1.HDF' # 有效文件路径参考./data-GIIRS/specificLonLat.csv
    # 这里缺少一步根据geodesc判断坐标是否在文件中，目前暂时由read-hdf.py生成的有效路径./data-GIIRS/specificLonLat.csv代替
    geo_desc = [28.7, 41.5, 116.2, 129.0, 0.05] # lat_S, lat_N, lon_W, lon_E, step
    file = FY4B_GIIRS_L1(filename, geo_desc)
    Solar_Zenith_VIS_new = file.extract()
    print(Solar_Zenith_VIS_new)