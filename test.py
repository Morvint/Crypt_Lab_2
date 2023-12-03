import cv2
import numpy as np
from time import sleep

# def rev(var):
#     res = ''
#     for i in range (len(var) // 5):
#         # print(term[i*5:i*5+5])
#         res = var[i*5:i*5+5] + res
#     return res


# term = "wasd1wasd2dsaw3dsaw4"
# term = rev(term)
# # print(term)

# file_name_img = "img.png"
# img = cv2.imread(file_name_img)
# size = img.shape
# if(size[0] * size[1] % 8 != 0):
#     a = 8 - (size[0] % 8), 8 - (size[1] % 8)
#     if a[0] < a[1]:
#         dop = np.zeros((a[0], size[1], 3))
#         img = np.concatenate((img, dop), axis=0)
#     else:
#         dop = np.zeros((size[0], a[1], 3))
#         img = np.concatenate((img, dop), axis=1)
# cv2.imwrite(file_name_img, img)

# sleep(1)
# f = open('term.txt', 'w')
# f.write("0")
# f.close()

# sleep(2)
# with open('term.txt', 'r') as f:
#     r = f.read(1)
# print(r)
	
# sleep(2)
# f = open('term.txt', 'w')
# f.write("1")
# f.close()

# import tqdm
# import time
# for i in tqdm.tqdm(range(100)):
#     time.sleep(0.25)

# import numpy as np
# import cv2

# img = cv2.imread("C:/Users/tv_20/Downloads/term.png")

# red_chanel = img[:, :, 2]
# print("Красная компонента -->", red_chanel)
# green_chanel = img[:, :, 1]
# print("Зеленая компонента -->", green_chanel)
# blue_chanel = img[:, :, 0]
# print("Синяя компонента   -->", blue_chanel)

# # np.save('saves_data', red_chanel)

# from art import tprint

# tprint("Term")

# import enquiries
# options = ['Do Something 1', 'Do Something 2', 'Do Something 3']
# choice = enquiries.choose('Choose one of these options: ', options)
# print(choice)

# from pick import pick

# title = 'Please choose your favorite programming language: '
# options = ['Java', 'JavaScript', 'Python', 'PHP', 'C++', 'Erlang', 'Haskell']

# option, index = pick(options, title, indicator='=>', default_index=2)

import os
from simple_term_menu import TerminalMenu


def main():
    fruits = ["[a] apple", "[b] banana", "[o] orange"]
    terminal_menu = TerminalMenu(fruits, title="Fruits")
    menu_entry_index = terminal_menu.show()


if __name__ == "__main__":
    main()