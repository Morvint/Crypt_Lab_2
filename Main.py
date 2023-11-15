import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

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
	Передается два массива NumPy размера"""
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

def round_key(key):
	"""Функция генерации ключей раундов"""
	key = ''.join(format(ord(x), '08b') for x in key)
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

def FEAL_encryption (text, key, i):
	"""Сам алгоритм FEAL"""
	if (i % 1000 == 0):
		print (i)
		
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
	text = ''.join(format(ord(x), '08b') for x in text)
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

def main():
	file_name = "C:/Prog/Python/Crypto_Lab_2/img.png"
	img = cv2.imread(file_name)
	# print(img.shape[0] * img.shape[1] * img.shape[2])
	print(img.shape)
	img = img.reshape(img.shape[0] * img.shape[1] * img.shape[2])
	img = img.reshape(277 * 277 * 3)
	img = ''.join(format(x, '08b') for x in img)
	print(len(img))

	# cv2.imshow("Binary", img)
	# cv2.waitKey(0)
	# img = img[:, ::, ::-1]
	# plt.imshow(img)
	# plt.show()
	# print(img)

	# cv2.imwrite("C:/Prog/Python/Crypto_Lab_2/newimg.png", img)





	key = round_key("qwertyst")

	# message = "wasdwasd"
	# message = ''.join(format(ord(x), '08b') for x in message)
	message = img


	while (round(len(message) % 64) != 0):
		message += '0'
	print(len(message))
	# img = np.array(list(message[0:1841496]))
	# print(img.shape)

	# for i in range(0, len(message) // 64):
	#     crypt = ''.join(FEAL_encryption(message[i*64:i*64+64], key))

	# print(message[28773*64:28773*64+64])
	# print(len(message) // 64)
	# crypt += ''.join(FEAL_encryption(message[i*64:i*64+64], key, i) for i in range(0, len(message) // 64))

	crypt = ''
	for i in range(len(message) // 64):
		text = FEAL_encryption(message[i*64:i*64+64], key, i)
		# print(message[i*64:i*64+64])
		crypt += ''.join(text)
	img = crypt[0:277*277*3*8]
	c = np.zeros([0])
	for i in range(len(img) // 8):
		if(i % 25000 == 0):
			print(i)
		j = int(img[i*8:i*8+8], 2)
		c = np.concatenate((c, j), axis=None)
		# с = np.append(c, j, axis=None)
	print(c.shape)
	img = c
	
	
	# print(len(img))
	img = img.reshape(277, 277, 3)
	cv2.imwrite("C:/Prog/Python/Crypto_Lab_2/newimg.png", img)

	# crypt = FEAL_encryption(message, key)
	# crypt = ''.join(crypt)
	# text = ''.join(chr(int(crypt[i*8:i*8+8], 2)) for i in range(len(crypt) // 8))
	"""
	decrypt = FEAL_deencryption(text, key)
	decrypt = ''.join(decrypt)
	text = ''.join(chr(int(decrypt[i*8:i*8+8], 2)) for i in range(len(decrypt) // 8))"""
	# print(text)

	# i = np.zeros([0])
	# j = np.zeros([1])
	# c = np.append(i, j, axis=None)
	# c = np.append(c, j, axis=None)
	# c = np.append(c, 97, axis=None)

	# print(c)


if __name__ == "__main__":
	main()