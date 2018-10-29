import numpy as np
import matplotlib.pyplot as plt
for i in range(3):
    poisson = np.random.poisson(4,1000)
    plt.ylabel("P(X) ------>")
    plt.xlabel("x ------>")
    plt.hist(poisson, density=True)
    np.savetxt("Poisson_Dataset" + str(i), poisson)
    plt.savefig('Phistogram'+str(i)+'.png')
    plt.show()
