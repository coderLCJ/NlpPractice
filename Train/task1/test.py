import numpy as np
import matplotlib.pyplot as plt
def H3(x):
    return np.where(x<0,0,(np.where(x<1,x,(np.where(x<2,2-x,0)))))
x = np.linspace(-3,5,1000)
y = H3(x)
plt.title('Plotting hat func in this plot')
plt.plot(x,y,'b-')
plt.show()
plt.savefig('picture/step4/fig4.png')