import os
import sys

if sys.platform == "win32":
    print("Running on Windows")
if sys.platform == "linux" or sys.platform == "linux2":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("Running on Linux")
if sys.platform == "darwin":
    print("Running on Mac")
