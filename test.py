import cv2
import numpy as np

# def rev(var):
#     res = ''
#     for i in range (len(var) // 5):
#         # print(term[i*5:i*5+5])
#         res = var[i*5:i*5+5] + res
#     return res


# term = "wasd1wasd2dsaw3dsaw4"
# term = rev(term)
# # print(term)

file_name_img = "img.png"
img = cv2.imread(file_name_img)
size = img.shape
if(size[0] * size[1] % 8 != 0):
    a = 8 - (size[0] % 8), 8 - (size[1] % 8)
    if a[0] < a[1]:
        dop = np.zeros((a[0], size[1], 3))
        img = np.concatenate((img, dop), axis=0)
    else:
        dop = np.zeros((size[0], a[1], 3))
        img = np.concatenate((img, dop), axis=1)
cv2.imwrite(file_name_img, img)