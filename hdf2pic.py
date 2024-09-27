import os
import h5py
import numpy as np
from numpy import deg2rad, rad2deg, arctan, arcsin, tan, sqrt, cos, sin
from numpy import *
import cv2
from datetime import *
import time
import math
from netCDF4 import Dataset
from skimage import exposure

ea = 6378.137  # 地球的半长轴[km]
eb = 6356.7523  # 地球的短半轴[km]
h = 42164  # 地心到卫星质心的距离[km]
λD = deg2rad(105.0)  # 卫星星下点所在经度

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
def latlon2linecolumn(lat, lon, resolution='4000M'):
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






def hdf2np(hdf_path,geo_range):
    """
    input:hdf file path
    output:numpy array 14*1000*1200,dtype=float32  || None

    """

    lat_S, lat_N, lon_W, lon_E, step = [int(x*10000) for x in geo_range] 
    lat = np.arange(lat_N-step, lat_S-step, -step)/10000
    lon = np.arange(lon_W, lon_E, step)/10000
    lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    line_org, column_org = latlon2linecolumn(lat_mesh, lon_mesh)

    res = []
    try:
        with h5py.File(hdf_path, 'r') as f:
            line = np.rint(line_org)-f.attrs['Begin Line Number'][()][0]
            line[line<=0]=0
            line = line.astype(np.uint16)
            column = np.rint(column_org).astype(np.uint16)

            for k in range(1,16):
                nom_name = 'NOMChannel'+'0'*(2-len(str(k)))+str(k)
                print(nom_name)
                cal_name = 'CALChannel'+'0'*(2-len(str(k)))+str(k) 
                print(cal_name)
                try:
                    nom = f["Data"][nom_name][()]
                    cal = f["Calibration"][cal_name][()]

                except:

                    with open('./error.log','a') as f:
                        f.write(hdf_path+' Channel:'+str(k)+'\n')
                    return 

                channel = nom[line, column]
                CALChannel = cal.astype(np.float32)


                CALChannel = np.append(CALChannel, 0)
                channel[channel >= 65534] = 4096

                res.append(CALChannel[channel])

            # save
            return np.array(res,dtype=np.float32)


    except:
        with open('./error.log','a') as f:
            f.write('Can not open:'+hdf_path+'\n')

        return None
   
def rec_datetime(path):
    mode = path.split('_')[3]
    
    if mode == 'DISK':
        datetime_1 = path.split('_')[-3]
        min = datetime_1[10:12]
        if int(min)>=0 and int(min)<30:
            min = '00'
        else:
            min = '30'
        datetime = datetime_1[:8]+'_'+datetime_1[8:10]+min
    
    else:
        datetime_1 = path.split('_')[-4]
        min = datetime_1[10:12]
        if int(min)==30:
            datetime = datetime_1[:8]+'_'+datetime_1[8:10]+'30'
        else:
            datetime = None
    
    return datetime

def make_RGB(np_out):
    # Forest_exist = (np_out[2] - np_out[1])/(np_out[2] + np_out[1])
    Forest_exist = np_out[2] / np_out[1]
    # print(Forest_exist[255])
    # tmp = np.zeros((256,256))
    temp = (Forest_exist > 1.5)
    channel2, channel1 = np_out[1].copy(), np_out[2].copy()
    channel1[temp] = (0.55 * channel1 + 0.45 * channel2)[temp]

    b = np_out[0, :, :]
    b = (b - np.min(b)) / (np.max(b) - np.min(b)) * 255
    g = channel1
    g = (g - np.min(g)) / (np.max(g) - np.min(g)) * 255
    r = channel2
    r = (r - np.min(r)) / (np.max(r) - np.min(r)) * 230
    changecol_pic = cv2.merge([b, g, r])
    return changecol_pic.astype(np.uint8)

def make_night_RGB(np_out):
    b = np.zeros(np_out[0,:,:].shape)
    g = np.zeros(np_out[0,:,:].shape)
    r = np.zeros(np_out[0,:,:].shape)
    night_tc = cv2.merge([b, g, r])
    # cv2.imwrite("./RGB4.png",night_tc)
    return night_tc

def make_nc_RGB(np_out):
    b = np_out[1, :, :]
    b = (b - np.min(b)) / (np.max(b) - np.min(b)) * 255
    g = np_out[2, :, :]
    g = (g - np.min(g)) / (np.max(g) - np.min(g)) * 255
    r = np_out[12, :, :]
    r = (r - np.min(r)) / (np.max(r) - np.min(r)) * 255
    FY4_nc = cv2.merge([b, g, r])
    return FY4_nc.astype(np.uint8)


def make_night_nc_RGB(np_out):
    # 做夜间海雾图
    channels = ['R', 'G', 'B']
    img_dic, range_dic, channel_dic, gamma_dic = {}, {}, {}, {}
    list1 = [13, 12, 12]
    list2 = [12, 8, -1]

    for i in range(3):
        ch = channels[i]
        # range_dic = { 'R' : (2, -4) , 'G' : (10,0) , 'B' : (293,243)}
        # range_dic = {'R': (6, -14), 'G': (20, -33), 'B': (290, 205)}
        range_dic = {'R': (2, -4), 'G': (9, 0), 'B': (293, 243)}
        channel_dic = {'R': (13, 12), 'G': (12, 8), 'B': (12, 0)}
        gamma_dic = {'R': "1.0", 'G': "1.0", 'B': "1.0"}
        img_a = np_out[list1[i] - 1, :, :]
        img_b = np.zeros(img_a.shape)  # TODO: add dtype attribute
        if list2[i] != -1:
            img_b = np_out[list2[i] - 1, :, :]
        img_dic[ch] = substract(img_a, img_b, range_dic[ch], channel_dic[ch], ch)

        for key, value in img_dic.items():
            if value is None:
                print("value is None")

    merged_img = img_merge(img_dic, gamma_dic)
    return merged_img


def img_merge(img_dic, gamma_dic):
    """
    Input three images (R, G, B) and gamma
    Merge them and return.
    """
    rgb = []
    for key in img_dic:
        rgb.append(img_dic[key])
    gamma = []
    for key in gamma_dic:
        gamma.append(gamma_dic[key])

    # gamma变换
    gamma_img_r = exposure.adjust_gamma(rgb[0], float(gamma[0]))
    gamma_img_g = exposure.adjust_gamma(rgb[1], float(gamma[1]))
    gamma_img_b = exposure.adjust_gamma(rgb[2], float(gamma[2]))
    cv2.imwrite('gamma_r.png', gamma_img_r)
    img = cv2.merge([gamma_img_b, gamma_img_g, gamma_img_r])

    return img


def substract(img_a, img_b, ranges, channels, channel):
    """
    return A - B with range (tuple)
    """
    maximum, minimum = ranges
    maximum = int(maximum)
    minimum = int(minimum)
    img_a = img_a.astype(np.int32)
    img_b = img_b.astype(np.int32)
    # img_a = recover_orgin_data(img_a, channels[0])
    # img_b = recover_orgin_data(img_b, channels[1])

    img = img_a - img_b
    img[img >= maximum] = maximum
    img[img <= minimum] = minimum
    img = cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)

    return img

if __name__ == "__main__":
    
    region = [28.7, 41.5, 116.2, 129.0, 0.0125]
    
    paths = [x for x in os.listdir('./data-fog/') if x.split('.')[-1]=='hdf' or x.split('.')[-1]=='HDF']
    paths = [x for x in paths if x[:4]=='FY4B']
    for path in paths:
        pathdir = './data-fog/' + path
        print(pathdir)
        datetime = rec_datetime(path)
        print(datetime)
        if datetime != None:
            npy = hdf2np(pathdir, region)
            # print(npy.dtype)
            print(npy.shape)
            np.save('./data-fog/FY4B_'+datetime+'.npy', npy)
            nc = make_nc_RGB(npy)
            rgb = make_RGB(npy)
            
            cv2.imwrite('./data-fog/FY4B_'+ 'RGB_' +datetime+".png", rgb)
            cv2.imwrite('./data-fog/FY4B_'+ 'NC_' +datetime+".png", nc)