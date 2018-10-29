import numpy as np
import matplotlib.pyplot as plt
for i in range(3):
    normal = np.random.normal(0,4,1000)
    plt.ylabel("N(X) ------>")
    plt.xlabel("x ------>")
    plt.hist(normal, density=True)
    np.savetxt("Normal_Dataset" + str(i), normal)
    plt.savefig('Nhistogram'+str(i)+'.png')
    plt.show()
