import matplotlib.pyplot as plt
import csv

header = ['num_vertices', 'density', 'random_seed', 'c_optimal', 'neurosa_res', 'iter2sota', 'iter2solution', 't2sota', 't2solution']

def visualize_matrix(H):
    plt.imshow(H, cmap='Greys', interpolation='none')
    plt.show()

def write_res(res_dir_file, res_list):
    with open(res_dir_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for res in res_list:
            writer.writerow(res)