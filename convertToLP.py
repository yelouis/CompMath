# https://github.com/hasan-kamal/Linear-Program-Solvers/blob/master/utils.py
def network_flow_to_std_LP(G, s, t):
	"""
	This function converts a given network (max-)flow problem into a std form LP min (c.T * x) s.t. Ax = b, x >= 0
	Parameters:
		G (np array): weighted adjacency matrix of graph where G[i][j] denotes capacity of directed edge i->j
		s 	   (int): source vertex, 0-based indexing
		t 	   (int): sink vertex, 0-based indexing
	Returns:
		c, A, b (np arrays): specify the LP in std form
		id_to_edge 	  (map): mapping from edge ids to edges
	"""

	N = len(G)
	edges = []
	id_to_edge = {}
	edge_to_id = {}
	num_edges = 0

	# dictionaries that hold list of in and out nbours of each node
	in_nbours = {}
	out_nbours = {}

	for i in range(N):
		for j in range(N):
			# skip this edge if not present
			if G[i][j] == 0:
				continue

			if i not in out_nbours:
				out_nbours[i] = []
			out_nbours[i].append(j)

			if j not in in_nbours:
				in_nbours[j] = []
			in_nbours[j].append(i)

			# edge i -> j exists
			edge = (i, j)
			edges.append(edge)
			id_to_edge[num_edges] = edge
			edge_to_id[edge] = num_edges
			num_edges += 1

	# we have a slack variable also corresponding to each edge (needed for capacity constraint)
	# first we embed flow variables, then slack variables
	num_variables = 2 * num_edges

	# create c
	c = np.zeros(shape=(num_variables, ))
	for i in range(num_edges):
		edge = id_to_edge[i]
		if edge[0] == s:
			# this edge exits s
			# so this variable should be present in c
			c[i] = -1.0

	# create A, b matrices
	# first we embed num_edges # of capacity constraints
	# then we embed N - 2 conservation constraints
	num_rows = num_edges + N - 2
	A = np.zeros(shape=(num_rows, num_variables))
	b = np.zeros(shape=(num_rows, ))

	# embed num_edges # of capacity constraints
	for i in range(num_edges):
		edge = id_to_edge[i]
		A[i, i] = 1.0 # add flow variable to constraint
		A[i, i + num_edges] = 1.0 # add slack variable to constrain
		b[i] = G[edge[0]][edge[1]]

	# we embed N - 2 conservation constraints
	num_non_terminal_nodes = 0
	for i in range(N):
		if i == s or i == t:
			continue

		# out neighbours
		for out_nbour in out_nbours[i]:
			A[num_edges + num_non_terminal_nodes, edge_to_id[(i, out_nbour)]] = 1.0

		# in neighbours
		for in_nbour in in_nbours[i]:
			A[num_edges + num_non_terminal_nodes, edge_to_id[(in_nbour, i)]] = -1.0

		# handle self-loop
		if (i, i) in edge_to_id:
			A[num_edges + num_non_terminal_nodes, edge_to_id[(i, i)]] = 0

		b[num_edges + num_non_terminal_nodes] = 0.0
		num_non_terminal_nodes += 1

	return c, A, b, id_to_edge
