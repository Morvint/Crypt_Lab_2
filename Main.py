import numpy as np
import cv2
import tqdm
from art import tprint
import random
import matplotlib.pyplot as plt

def RGB_chanels(img):
    red_chanel = img[:, :, 2]
    green_chanel = img[:, :, 1]
    blue_chanel = img[:, :, 0]

    red_chanel = red_chanel.reshape(red_chanel.shape[0] * red_chanel.shape[1])
    green_chanel = green_chanel.reshape(green_chanel.shape[0] * green_chanel.shape[1])
    blue_chanel = blue_chanel.reshape(blue_chanel.shape[0] * blue_chanel.shape[1])

    return red_chanel, green_chanel, blue_chanel

def foo(path):
    file_path = path
    img = cv2.imread(file_path)
    size = img.shape
    if(size[0] * size[1] % 8 != 0):
        a = 8 - (size[0] % 8), 8 - (size[1] % 8)
        if a[0] < a[1]:
            dop = np.zeros((a[0], size[1], 3))
            img = np.concatenate((img, dop), axis=0)
        else:
            dop = np.zeros((size[0], a[1], 3))
            img = np.concatenate((img, dop), axis=1)
    cv2.imwrite(file_path, img)

def rev(var):
    """ 
        Функция инвертирования строки
        Передается строка
    """
    res=''
    for i in range(len(var)-1,-1,-1):
        res+=var[i]
    return res

def XOR (a0, a1):
    """Исключающее ИЛИ;
    Передается два массива NumPy одинакового размера"""

    a2 = np.empty_like(a0)

    for i in range (0, len(a0)):
        if(a0[i] == a1[i]):
            a2[i] = '0'
        else:
            a2[i] = '1'

    # print (a2)
    return(a2)

def S0(a0, a1):
    """Циклический сдвиг влево на два бита ((a + b) mod 256);
    Передается два массива NumPy размера 8"""
    a0 = ''.join(a0)
    a1 = ''.join(a1)
    
    a0 = int(a0, 2)
    a1 = int(a1, 2)
    # print(a0, a1)

    a2 = (round((a0 + a1) % 256))
    a2 = format(a2, '08b')
    a2 = a2[-2:] + a2[:-2]
    a2 = np.array(list(a2))
    # print ("", a2)
    return a2

def S1(a0, a1):
    """Циклический сдвиг влево на два бита ((a + b + 1) mod 256);
    Передается два массива NumPy размера 8"""

    a0 = ''.join(a0)
    a1 = ''.join(a1)
    
    a0 = int(a0, 2)
    a1 = int(a1, 2)
    # print(a0, a1)

    a2 = (round((a0 + a1 + 1) % 256))
    # print (a2)
    a2 = format(a2, '08b')
    a2 = a2[-2:] + a2[:-2]
    a2 = np.array(list(a2))
    # print (a2)
    return a2

def F_k(A, B):
    """Функция F для генерации ключей раундов;
    Передается два массива NumPy размера 32"""
    A = A.reshape(4, A.size // 4)
    B = B.reshape(4, B.size // 4)
    C = np.empty_like(A)
    # print(A)
    # print(B)

    A[1] = XOR(A[0], A[1])
    A[2] = XOR(A[2], A[3])
    C[1] = XOR(B[0], A[2])
    C[1] = S1(A[1], C[1])
    C[2] = XOR(B[1], C[1])
    C[2] = S0(A[2], C[2])
    C[0] = XOR(B[2], C[1])
    C[0] = S0(A[0], C[0])
    C[3] = XOR(B[3], C[2])
    C[3] = S1(C[3], A[3])

    C = C.reshape(32)
    # print(C, C.shape)
    return C

def get_key(key):
    key = np.array(list(key))
    key = key.reshape(8, 16)
    return key

def round_key(key):
    """Функция генерации ключей раундов"""
    if (len(key) == 8):
        key = ''.join(format(ord(x), '08b') for x in key)
        print("term")
    key = np.array(list(key))
    key = key.reshape(2, key.size // 2)
    L, R = key[0], key[1]

    """шаг 1"""
    D = R
    key0 = F_k(L, R)
    D = XOR(L, key0)
    L = R
    R = key0
    key_r = np.copy(key0.reshape(2, 16))

    """шаг 2"""
    key1 = F_k(L, R)
    D = XOR(L, key1)
    L = R
    R = key1
    key_r = np.concatenate((key_r, key1.reshape(2, 16)), axis=0)

    """шаг 3"""
    key2 = F_k(L, R)
    D = XOR(L, key2)
    L = R
    R = key2
    key_r = np.concatenate((key_r, key2.reshape(2, 16)), axis=0)

    """шаг 4"""
    key3 = F_k(L, R)
    key_r = np.concatenate((key_r, key3.reshape(2, 16)), axis=0)

    
    # print(key_r, key_r.shape)
    # print(key_r.shape)
    return key_r

def F (text, key):
    """Функция F для алгоритма FEAL"""
    text = text.reshape(4, 8)
    key = key.reshape(2, 8)
    C = np.empty_like(text)
    # print(text)
    # print(key)
    
    C[1] = XOR(key[0], text[1])
    C[2] = XOR(key[1], text[2])
    C[1] = XOR(text[0], C[1])
    C[2] = XOR(text[3], C[2])

    C[1] = S1(C[2], C[1])
    C[2] = S0(C[1], C[2])
    C[0] = S0(C[1], text[0])
    C[3] = S1(C[2], text[3])

    C = C.reshape(32)
    # print(C)
    return C

def FEAL_encryption (text, key):
    """Сам алгоритм FEAL"""
        
    # print("t --> ", text)
    text = np.array(list(text))
    # print(text.shape)
    
    text = XOR(text, key[4:8].reshape(64))
    text = text.reshape(2, 32)
    L, R = text[0], text[1]

    """шаг 1"""
    R = XOR(R, L)
    D = F(R, key[0])
    L = XOR(L, D)
    # R = XOR(L, R)
    C = L
    L = R
    R = C
    
    """шаг 2"""
    D = F(R, key[1])
    L = XOR(L, D)
    C = L
    L = R
    R = C

    """шаг 3"""
    D = F(R, key[2])
    L = XOR(L, D)
    C = L
    L = R
    R = C

    """шаг 4"""
    D = F(R, key[3])
    L = XOR(L, D)
    R = XOR(R, L)

    text[0], text[1] = L, R
    text = text.reshape(64)
    text = XOR(text, key[4:8].reshape(64))

    return text

def FEAL_deencryption (text, key):
    """Сам алгоритм FEAL, вернее дешифратор"""
    # text = ''.join(format(ord(x), '08b') for x in text)
    text = np.array(list(text))

    text = XOR(text, key[4:8].reshape(64))
    text = text.reshape(2, 32)
    L, R = text[0], text[1]

    """шаг 1"""
    R = XOR(R, L)
    D = F(R, key[3])
    L = XOR(L, D)
    # R = XOR(L, R)
    C = L
    L = R
    R = C

    """шаг 2"""
    D = F(R, key[2])
    L = XOR(L, D)
    C = L
    L = R
    R = C

    """шаг 3"""
    D = F(R, key[1])
    L = XOR(L, D)
    C = L
    L = R
    R = C

    """шаг 4"""
    D = F(R, key[0])
    L = XOR(L, D)
    R = XOR(R, L)

    text[0], text[1] = L, R
    text = text.reshape(64)
    text = XOR(text, key[4:8].reshape(64))

    return text

def FEAL_encryption_term (text, key, chipher):
    """Сам алгоритм FEAL"""
        
    # print("t --> ", text)
    text = np.array(list(text))
    chipher = np.array(list(chipher))
    # print(text.shape)

    text = XOR(text, chipher)
    
    text = XOR(text, key[4:8].reshape(64))
    text = text.reshape(2, 32)
    L, R = text[0], text[1]

    """шаг 1"""
    R = XOR(R, L)
    D = F(R, key[0])
    L = XOR(L, D)
    # R = XOR(L, R)
    C = L
    L = R
    R = C
    
    """шаг 2"""
    D = F(R, key[1])
    L = XOR(L, D)
    C = L
    L = R
    R = C

    """шаг 3"""
    D = F(R, key[2])
    L = XOR(L, D)
    C = L
    L = R
    R = C

    """шаг 4"""
    D = F(R, key[3])
    L = XOR(L, D)
    R = XOR(R, L)

    text[0], text[1] = L, R
    text = text.reshape(64)
    text = XOR(text, key[4:8].reshape(64))

    return text

def FEAL_deencryption_term (text, key, chipher):
    """Сам алгоритм FEAL, вернее дешифратор"""
    # text = ''.join(format(ord(x), '08b') for x in text)
    text = rev(text)
    chipher = rev(chipher)
    text = np.array(list(text))
    chipher = np.array(list(chipher))

    

    text = XOR(text, key[4:8].reshape(64))
    text = text.reshape(2, 32)
    L, R = text[0], text[1]

    """шаг 1"""
    R = XOR(R, L)
    D = F(R, key[3])
    L = XOR(L, D)
    # R = XOR(L, R)
    C = L
    L = R
    R = C

    """шаг 2"""
    D = F(R, key[2])
    L = XOR(L, D)
    C = L
    L = R
    R = C

    """шаг 3"""
    D = F(R, key[1])
    L = XOR(L, D)
    C = L
    L = R
    R = C

    """шаг 4"""
    D = F(R, key[0])
    L = XOR(L, D)
    R = XOR(R, L)

    text[0], text[1] = L, R
    text = text.reshape(64)
    text = XOR(text, key[4:8].reshape(64))

    text = XOR(text, chipher)

    return text

def main():
    tprint("Programm Run")

    file_name_img = "img.png"
    file_name_newimg = "newimg.png"
    file_name_decryptimg = "decryptimg.png"
    file_name_newimg_copy = "newimg_copy.png"

    # Генерация ключа
    key = round_key("zxcvasdf")

    # Создание вектора инициализации для режима CBC
    init_vector = "wasdwasd"
    init_vector = ''.join(format(ord(x), '08b') for x in init_vector)

    print("Enter [Y] to run encrypt -->", end=" ")
    flag = input()
    print("Enter [Y] to run decrypt -->", end=" ")
    flag1 = input()

    if flag == "Y":
        print("Encrypt Run")

        # Окрытие картинки и преобразование в необходимый формат и получение RGB компонент
        foo(file_name_img)
        img = cv2.imread(file_name_img)
        red_chanel_before, green_chanel_before, blue_chanel_before = RGB_chanels(img)
        img_shape = np.array(img.shape)
        img = img.reshape(img_shape[0] * img_shape[1] * img_shape[2])
        img = ''.join(format(x, '08b') for x in img)

        message = img

        # Шифровка в режиме работы CBC
        crypt0 = ''
        for i in tqdm.tqdm(range(len(message) // 64)):
            if i < 1:
                text = FEAL_encryption_term(message[i*64:i*64+64], key, init_vector)
                crypt0 += ''.join(text)
            else:
                text = FEAL_encryption_term(message[i*64:i*64+64], key, crypt0[(i - 1)*64:(i - 1)*64+64])
                crypt0 += ''.join(text)

        ones = crypt0.count("1")
        print("Eдиниц выходном потоке -- >", ones)
        zeros = crypt0.count("0")
        print("Нулей выходном потоке -- >", zeros)
        print("Отношение 1 к длине -->", ones / len(crypt0))
        print("Отношение 0 к длине -->", zeros / len(crypt0))

        # Отсечение дополнительных бит
        # img = crypt0[0:img_shape[0] * img_shape[1] * img_shape[2] * 8]
        img = crypt0

        # Преобразование строки в форму подходящую для картинки
        c = []
        for i in range(len(img) // 8):
            j = int(img[i*8:i*8+8], 2)
            c.append(j)
        c = np.array(c)
        img = c
        img = img.reshape(img_shape[0], img_shape[1], img_shape[2])
        
        # Сохраниение картинки
        cv2.imwrite(file_name_newimg, img)
        red_chanel_after, green_chanel_after, blue_chanel_after = RGB_chanels(img)
        print("Корреляция RG -->", np.corrcoef(red_chanel_before, green_chanel_after)[0, 1])
        print("Корреляция GB -->", np.corrcoef(green_chanel_before, blue_chanel_after)[0, 1])
        print("Корреляция BR -->", np.corrcoef(blue_chanel_before, red_chanel_after)[0, 1])
        print("Encrypt successfully")


        print("Decrypt Run")

        # Извлечение картики для дешифрации
        crypt = cv2.imread(file_name_newimg)
        crypt_shape = np.array(crypt.shape)
        crypt = crypt.reshape(crypt_shape[0] * crypt_shape[1] * crypt_shape[2])
        crypt = ''.join(format(x, '08b') for x in crypt)

        # дешифровка в режиме работы CBC
        decrypt = ''
        # Инвертирование полученной строки
        crypt = rev(crypt)
        for i in tqdm.tqdm(range(len(crypt) // 64)):
            if i < (len(crypt) // 64) - 1:
                text = FEAL_deencryption_term(crypt[i*64:i*64+64], key, crypt[(i + 1)*64:(i + 1)*64+64])
                # Повторное инвертирование блока
                text = rev(text)
                decrypt += ''.join(text)
            else:
                text = FEAL_deencryption_term(crypt[i*64:i*64+64], key, init_vector)
                # Повторное инвертирование блока
                text = rev(text)
                decrypt += ''.join(text)
        decrypt = rev(decrypt)

        # Отсечение дополнительных бит
        img = decrypt[0:crypt_shape[0] * crypt_shape[1] * crypt_shape[2] * 8]

        # Преобразование строки в форму подходящую для картинки
        z = []
        for i in range(len(img) // 8):
            j = int(img[i*8:i*8+8], 2)
            z.append(j)
        z = np.array(z)
        img = z
        img = img.reshape(crypt_shape[0], crypt_shape[1], crypt_shape[2])

        # Сохраниение картинки
        cv2.imwrite(file_name_decryptimg, img)
        print("Decrypt successfully")

def main1():
	
    file_name1 = "img.png"
    file_name2 = "newimg.png"
    file_name3 = "decryptimg.png"
    file_name4 = "newimg_copy.png"

    foo(file_name1)
    img = cv2.imread(file_name1)
    img_shape = np.array(img.shape)
    img = img.reshape(img_shape[0] * img_shape[1] * img_shape[2])
    img = ''.join(format(x, '08b') for x in img)
    # print(len(img))


    key = round_key("zxcvasdf")

    # message = "wasdwasdwasdwasd"
    # message = ''.join(format(ord(x), '08b') for x in message)
    message = img


    while (round(len(message) % 64) != 0):
        message += '0'
        print("1")

    crypt0 = ''
    for i in tqdm.tqdm(range(len(message) // 64)):
        text = FEAL_encryption(message[i*64:i*64+64], key)
        crypt0 += ''.join(text)
    img = crypt0[0:img_shape[0] * img_shape[1] * img_shape[2] * 8]

    c = []
    for i in range(len(img) // 8):
        j = int(img[i*8:i*8+8], 2)
        c.append(j)
    c = np.array(c)
    img = c
    
    print("Encrypt successfully")
    img = img.reshape(img_shape[0], img_shape[1], img_shape[2])
    cv2.imwrite(file_name2, img)




    foo(file_name2)
    crypt = cv2.imread(file_name2)
    crypt_shape = np.array(crypt.shape)
    crypt = crypt.reshape(crypt_shape[0] * crypt_shape[1] * crypt_shape[2])
    crypt = ''.join(format(x, '08b') for x in crypt)
    # print(len(crypt))

    decrypt = ''
    for i in tqdm.tqdm(range(len(crypt) // 64)):
        text = FEAL_deencryption(crypt[i*64:i*64+64], key)
        decrypt += ''.join(text)
    img = decrypt[0:img_shape[0] * img_shape[1] * img_shape[2] * 8]

    z = []
    for i in range(len(img) // 8):
        j = int(img[i*8:i*8+8], 2)
        z.append(j)
    z = np.array(z)
    # print(z.shape)
    img = z

    print("Decrypt successfully")
    img = img.reshape(img_shape[0], img_shape[1], img_shape[2])
    cv2.imwrite(file_name3, img)
    
def diff_letters(a,b):
    return float(sum ( a[i] != b[i] for i in range(len(a)) ))

def Lab3():
    key = ""
    while len(key) < 128:
        key += "1"
    key = get_key(key)
    # print(key)

    key = ""
    while len(key) < 128:
        key += "0"
    key = get_key(key)
    # print(key)

    key = "00001111001100110000111100110011000011110011001100001111001100110000111100110011000011110011001100001111001100110000111100110011"
    key = get_key(key)
    # print(key)

    key = "00001111000011110000111100001111000011110000111100001111000011110000111100001111000011110000111100001111000011110000111100001111"
    key = get_key(key)
    # print(key)

    key = ""
    while len(key) < 128:
        key += random.choice(['0', '1'])
    key = get_key(key)
    # print(key.shape)


    text = "wasdwasd"
    message = ''.join(format(ord(x), '08b') for x in text)
    print("Исходный текст      -->", message)

    crypt0 = ''
    for i in tqdm.tqdm(range(len(message) // 64)):
        text = FEAL_encryption(message[i*64:i*64+64], key)
        crypt0 += ''.join(text)
    print("Защифрованный текст -->", crypt0)

    list_term = ''
    for i in range(64):
        if crypt0[i] != message[i]:
            list_term += "*"
        else:
            list_term += "_"
    print(list_term)
    

    # print("\033[34m{}".format("text"))

    list_graf0 = []
    list_graf1 = []
    for i in range(64):
        crypt1 = crypt0[-(i + 1):] + crypt0[:-(i+1)]
        list_graf0.append(i + 1)
        list_graf1.append(diff_letters(crypt0, crypt1) / 64.0)
    # print(list_graf0)

    # print(list_graf1)
    plt.plot(list_graf0, list_graf1)

    list_siries = []
    for i in range(0,64):
        list_siries.append(0)

    count = 1
    for i in range(1, 64):
        if crypt0[i - 1] == crypt0[i]:
            count += 1
        else:
            list_siries[count] += 1
            count = 1
    print(list_siries)

    ones = crypt0.count("1")
    print("Eдиниц выходном потоке -- >", ones)
    print("Отношение 1 к длине -->", ones / len(crypt0))
    plt.show()
    
def RSLOS_generate(key):
    # key = key[-1:] + key[:-1]
    
    if key[9] == key[2]:
        # key = "0" + key[-9:]
        key = "0" + key[:-1]
    else:
        # key = "1" + key[-9:]
        key = "1" + key[:-1]

    return key

def Lab3_base():
    key = "1010101010"
    text = RSLOS_generate(key)

    count = 0
    while text != key:
        text = RSLOS_generate(text)
        count += 1
    print(count)

    

if __name__ == "__main__":
    # main()
    # main1()
    # Lab3()
    Lab3_base()