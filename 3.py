import numpy as np
import matplotlib.pyplot as plt
normal = np.random.normal(0, 4, 1000)
poisson = np.random.poisson(4, 1000)
data= []
for i in range(100):
    a=np.random.randint(0,1000)
    avg=(poisson[a]+normal[a])
    avg=avg/2
    data.append(avg)
plt.ylabel("y ------>")
plt.xlabel("x ------>")
np.savetxt("Dataset", data)
plt.hist(data,density=True)
plt.savefig("Avg.png")
plt.show()
