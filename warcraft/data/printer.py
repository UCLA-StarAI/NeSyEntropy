import numpy as np
f = np.load("warcraft_shortest_path/12x12/test_shortest_paths.npy")
for i in range(f.shape[0]):
    print(f[i])