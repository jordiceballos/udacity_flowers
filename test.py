import time
from timeit import default_timer as timer
import pandas as pd
import matplotlib.pyplot as plt


# x = range(10)
# y = range(10)

# fig = plt.figure()

# plt.subplot(2, 1, 1)
# plt.plot(x, y)

# plt.subplot(2, 1, 2)
# plt.plot(x, y)

# plt.show()





x = range(10)
y = range(10)

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8,5))

ax[0].plot(x, y)
ax[1].plot(x, y)

fig.suptitle('Main title') 
plt.show()