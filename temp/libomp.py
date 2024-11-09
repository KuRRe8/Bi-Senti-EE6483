import os

def find_files(filename, search_path):
    result = []
    for root, dir, files in os.walk(search_path):
        if filename in files:
            result.append(os.path.join(root, filename))
    return result

# 查找系统盘中的 libiomp5md.dll 文件
dll_files = find_files('libiomp5md.dll', 'C:\\')
for dll in dll_files:
    print(dll)