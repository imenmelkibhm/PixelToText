import sys
import os
from ctypes import *
from itertools import takewhile

# Load the dll

ocrdll = cdll.LoadLibrary("./libocr.so")
ocrdll.executeTask.restype = c_wchar_p
ocrdll.executeTask.argtypes = [c_wchar_p]
ret = ocrdll.executeTask("/opt/exe/PixelToText/Debug_images/zone0_0_thresh.jpg")
print 'end of script'
print ret



