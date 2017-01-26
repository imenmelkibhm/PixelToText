import sys
import os
from ctypes import *

# Load the dll
mydll = cdll.LoadLibrary("./libHello.so")
mydll.executeTask()

