from qubo_mis_generator import MISBenchmark

import warnings
warnings.filterwarnings('ignore') # ignore warning from netx

if __name__ == '__main__':
	benchmark = MISBenchmark('default_config.csv')
	
	# maximum supported workload size
	test_size_100 = 1000
	qubo_matrix = MISBenchmark.custom_graph(test_size_100, 1.0)

	test_size_1 = 1000
	qubo_matrix = MISBenchmark.custom_graph(test_size_1, 0.01)

	
	# benchmark
	for qubo_matrix, num_vertices, density, seed, c_optimal in benchmark.generate_problems():
		print("Problem: ", num_vertices, density, seed)

		# configure problem, deploy and solve
		# qubo_matrix is an (N x N) symmetric np array with -1 on diagonal and 2 at edges.