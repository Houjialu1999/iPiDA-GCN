import numpy as np

def normal_row(data):
    data1=data.transpose()
    data1=data1/(sum(data1))
    return data1.transpose()

Z=np.random.random((3,2))
a=normal_row(Z)
print(a)