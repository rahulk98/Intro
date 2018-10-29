

import numpy as np
import matplotlib.pyplot as plt
for i in range(3):
    normal = np.random.normal(0,3,1000)
    np.savetxt("Normal_Dataset" + str(i), normal)
    plt.ylabel("N(X)=x")
    plt.xlabel("x")
    plt.hist(normal, density=True)
    plt.savefig('Nhistogram'+str(i)+'.png')
    plt.show()
