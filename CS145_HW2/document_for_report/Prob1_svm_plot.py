import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    plt.close('all')
    x = np.linspace(-5, 5, 1000)
    x1 = [0.52,0.91,-1.48,0.01,-0.46,0.41,0.53,-1.21,-0.39,-0.96,2.46,3.05,2.2,1.89,4.51,3.06,3.16,2.05,2.34,2.94]
    x2 = [-1,0.32,1.23,1.44,-0.37,2.04,0.77,-1.1,0.96,0.08,2.59,2.87,3.04,2.64,-0.52,1.3,-0.56,1.54,0.72,0.13]
    y = [1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    p1 = []
    p2 = []
    n1 = []
    n2 = []
    for i in range(0,20):
        if y[i] == 1:
            p1.append(x1[i])
            p2.append(x2[i])
        else:
            n1.append(x1[i])
            n2.append(x2[i])
    print(p1)
    print(p2)
    plt.plot(x,(-1.87578*x+3.68118)/0.814692 )
    plt.scatter(p1,p2,c=u'b')
    plt.scatter(n1,n2,c=u'r')
    plt.xlim(-3,5)
    plt.ylim(-3,4)
    plt.show()
