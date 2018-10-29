import numpy as np
import matplotlib.pyplot as plt
poisson = np.random.poisson(3, 1000)
normal = np.random.normal(0, 3, 1000)
data= []
for i in range(100):
    a=np.random.randint(0,1000)
    avg=(p[a]+b[a])/2
    data.append(avg)
    plt.hist(data,density=True)
np.savetxt("Dataset", data)
plt.hist(data, density=True)
plt.savefig("Avg.png")
plt.show()
