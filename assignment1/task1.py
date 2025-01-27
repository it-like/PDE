#a)
values = [(2, 2), (3, 6), (4, 5), (5, 5), (6, 6)]
values = [[gi[0],gi[1]] for gi in values] 

x = [gi[0] for gi in values] 
y = [gi[1] for gi in values]
import matplotlib.pyplot as plt # Plotting library
plt.scatter(x,y,marker='o')
plt.axis([0,7,0,7])
plt.show()

#b)
import numpy as np # Mathematics library

plt.figure(figsize=[12,8])
xx = np.linspace(min(x), max(x), 50)
z = np.polyfit(x,y,4)
p = np.poly1d(z)
plt.plot(x,y,'.',xx,p(xx))
plt.savefig('assignment1/photos/task1.png')
plt.show()







