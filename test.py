import matplotlib.pyplot as plt
import numpy as np
import nnfs

nnfs.init()

from nnfs.datasets import spiral_data
X,y=spiral_data(samples=10,classes=3)
print(y)