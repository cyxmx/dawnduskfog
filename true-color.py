import h5py
import matplotlib.pyplot as plt


filename = f'./data-fog/FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20240709090000_20240709091459_4000M_V0001.HDF'
# 打开HDF文件
with h5py.File(filename, 'r') as hdf:
    # 假设HDF文件中只有一个数据集（dataset）
    data = list(hdf.items())[0][1]
    
    # 如果数据是索引图像数据，可能需要先将其转换为标准形状
    # data = data.reshape(shape)
    
    # 显示图像
    plt.imshow(data)
    plt.show()