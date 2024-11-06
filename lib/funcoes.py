import numpy as np

#=======================================================================
def cm_to_inch(value):
    return value/2.54
#=======================================================================
def rad_to_angle(value):
    return value*180/np.pi
#=======================================================================
def angle_to_rad(value):
    return value*np.pi/180
#=======================================================================
def filtro_espacial(img, kernel):
    img_temp = np.zeros_like(img, dtype=np.float16)

    lk = kernel.shape[0]
    ck = kernel.shape[1]

    llin = int(lk/2)
    lcol = int(ck/2)

    res = 0
    for i in range(llin, img.shape[0] - llin):
        for j in range(lcol, img.shape[1] - lcol):
            janela = img[(i-llin):(i+llin+1), (j-lcol):(j+lcol+1)]
            res = janela * kernel

            img_temp[i,j] = np.sum(res)

    return img_temp
#=======================================================================
def gaussian_kernel(size, k, sigma):
    kernel = np.zeros((size,size))
    
    llin = int(size/2) #supoe impar
    lcol = llin #e quadrado
    
    for m in range(-llin, llin+1):
        for n in range(-lcol, lcol+1):
            kernel[m+llin, n+lcol] = k*np.exp(-(m**2+n**2) / (2*sigma**2))
    
    kernel = 1 / np.sum(kernel) * kernel 
    
    return kernel