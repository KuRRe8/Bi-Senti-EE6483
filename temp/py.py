import os
import sys

a = 100
b = 101

def func():
    global a
    print(a)

func()

print(a)

sys.exit(0)