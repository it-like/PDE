import matplotlib.pyplot as plt
import numpy as np 

years = [1975, 1980, 1985, 1990]
west_eu = [72.8, 74.2, 75.2, 76.4]
east_eu = [70.2, 70.2, 70.3, 71.2]

areas = [west_eu, east_eu]

def build_graph(xs, ys, pol):
    #polyval = np.polynomial.polynomial.polyval(xs,ys,pol)
    steps = np.linspace(min(xs)-10, max(xs)+10, 500)
    z = np.polyfit(xs,ys,pol)
    p = np.poly1d(z)
    return xs, ys, steps, p

def plot_graphs(xs,ys,steps,p):
    plt.plot(xs,ys,'.',steps,p(steps))
    

plt.figure(figsize=[12,8])

for area in areas:
    xs, ys, steps, p = build_graph(years,area,3)
    plot_graphs(xs, ys, steps , p)

compare_year = 1970
west_eu_com = 71.8
east_eu_com = 69.6
plt.grid(True)

plt.scatter(compare_year, west_eu_com, color='darkorange')
plt.text(compare_year, west_eu_com+0.5, 'West Europe', color='darkorange')
plt.scatter(compare_year, east_eu_com, color='darkred')
plt.text(compare_year, east_eu_com+0.5, 'East Europe', color='darkred')
plt.savefig('assignment1/photos/task2_longer_range.png')
plt.show()
