# PART 1: Matplotlib graph using PyScript
import numpy as np
import matplotlib.pyplot as plt

x1 = np.random.rand(80)
y1 = np.random.rand(80)
plt.scatter(x1, y1, color='green', label='First set')

x2 = np.random.rand(80)
y2 = np.random.rand(80)
plt.title('MATPLOTLIB GRAPH using PyScript')
plt.xlabel('Great for data visualization')
plt.ylabel('Adaptable to most Machine Learning packages')
plt.scatter(x2, y2, color='olive', label='Second set')
plt.show()


# PART 2: Random Numpy Array
def test_button():
    n_array = [1, 2, 3, 4, 5, 6, 7, 8]
    n_random = [np.random.choice(n_array) for _ in range(6)]
    print(n_random)


test_button()
