def rev(var):
    res = ''
    for i in range (len(var) // 5):
        # print(term[i*5:i*5+5])
        res = var[i*5:i*5+5] + res
    return res


term = "wasd1wasd2dsaw3dsaw4"
term = rev(term)
print(term)

# message = ''
# for i in range (len(term) // 5):
#     print(term[i*5:i*5+5])
#     message = term[i*5:i*5+5] + message
# print(message)

