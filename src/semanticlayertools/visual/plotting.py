class Multilayer3D():
    """Plot multiplex network. 
    This solution is based on this StackOverflow answer:

    https://stackoverflow.com/questions/60392940/multi-layer-graph-in-networkx/60416989
    """
    def __init__(
        self, graphs, graphLabels, node_positions, edge_paths
    ):
        self.graphs: list = graphs
        self.graphLabels: list = graphLabels
        self.total_layers = len(graphs)
        self.positions: dict = node_positions
        self.edgePaths: dict = edge_paths
        self.node_to_community = {}
        self.community_to_color = {}
        self.node_color = {}
        
        if ax:
            self.ax = ax
        else:
            fig = plt.figure()
            self.ax = fig.add_subplot(111, projection='3d')
        
        self.get_nodes()
        self.get_edges_within_layers()
        self.get_edges_between_layers()

        # compute layout and plot
        self.get_node_positions()
        self.draw()
        
    def getCommunities(self):
        composed = self.graphs[0]
        for g0 in self.graphs[1:]:
            composed = nx.compose(composed, g0)     
        mg = sorted(
            louvain_communities(composed),
            key=lambda x: len(x),
            reverse=True
        )
        self.node_to_community = {}
        for x,y in enumerate(mg):
            for elem in y:
                self.node_to_community.update({elem:x})
        self.community_to_color = {x:cm.tab10.colors[x] for x in set(self.node_to_community.values()) if x < 10}

        for elem in set(self.node_to_community.values()):
            if elem >= 10:
                self.community_to_color.update({elem: (0.5,0.5,0.5)})

        self.node_color = {node: self.community_to_color[community_id] for node, community_id in self.node_to_community.items()}
        
    def get_nodes(self):
        """Construct an internal representation of nodes with the format (node ID, layer)."""
        self.nodes = []
        for z, g in enumerate(self.graphs):
            self.nodes.extend([(node, z) for node in g.nodes()])
        
    def get_edges_within_layers(self):
        """Remap edges in the individual layers to the internal representations of the node IDs."""
        self.edges_within_layers = []
        for z, g in enumerate(self.graphs):
            self.edges_within_layers.extend([((source, z), (target, z)) for source, target in g.edges()])

    def get_edges_between_layers(self):
        """Determine edges between layers. Nodes in subsequent layers are
        thought to be connected if they have the same ID."""
        self.edges_between_layers = []
        for z1, g in enumerate(self.graphs[:-1]):
            z2 = z1 + 1
            h = self.graphs[z2]
            shared_nodes = set(g.nodes()) & set(h.nodes())
            self.edges_between_layers.extend([((node, z1), (node, z2)) for node in shared_nodes])

    def get_node_positions(self, *args, **kwargs):
        """Get the node positions in the layered layout."""
        self.node_positions = dict()
        for z, g in enumerate(self.graphs):
            self.node_positions.update({(node, z) : (*self.positions[node], z) for node in g.nodes()})
            
    def draw_nodes(self, nodes, *args, **kwargs):
        x, y, z = zip(*[self.node_positions[node] for node in nodes])
        colors = [self.community_to_color[self.node_to_community[x[0]]] for x in nodes ] 
        self.ax.scatter(x, y, z, c=colors, *args, **kwargs)


    def draw_edges(self, edges, *args, **kwargs):
        segments = [(self.node_positions[source], self.node_positions[target]) for source, target in edges]
        line_collection = Line3DCollection(segments, *args, **kwargs)
        self.ax.add_collection3d(line_collection)

    def draw_edges_from_path(self, edges, *args, **kwargs):
        for edge in edges:
            layernr = edge[0][1]
            path = self.edgePaths[(edge[0][0],edge[1][0])]
            layerarray = np.array([layernr]*len(path))            
            extPath = np.concatenate((path,layerarray.reshape((len(path),1)),), axis=1)
            segment = []
            for idx, elem in enumerate(extPath):
                if idx + 1 < len(extPath):
                    segment.append((extPath[idx], extPath[idx +1]))
            line_collection = Line3DCollection(segment, *args, **kwargs)
            self.ax.add_collection3d(line_collection)

    def get_extent(self, pad=0.1):
        xyz = np.array(list(self.node_positions.values()))
        xmin, ymin, _ = np.min(xyz, axis=0)
        xmax, ymax, _ = np.max(xyz, axis=0)
        dx = xmax - xmin
        dy = ymax - ymin
        return (xmin - pad * dx, xmax + pad * dx), \
            (ymin - pad * dy, ymax + pad * dy)

    def draw_plane(self, z, *args, **kwargs):
        (xmin, xmax), (ymin, ymax) = self.get_extent(pad=0.1)
        u = np.linspace(xmin, xmax, 10)
        v = np.linspace(ymin, ymax, 10)
        U, V = np.meshgrid(u ,v)
        W = z * np.ones_like(U)
        self.ax.plot_surface(U, V, W, *args, **kwargs)

    def draw(self):
        
        self.getCommunities()
        self.draw_edges_from_path(self.edges_within_layers,  color='k', alpha=0.05, linestyle='-', zorder=2)
        self.draw_edges(self.edges_between_layers, color='k', alpha=0.3, linestyle='dotted', zorder=2)

        for z in range(self.total_layers):
            self.draw_plane(z, alpha=0.1, zorder=1)
            self.draw_nodes([node for node in self.nodes if node[1]==z], s=100, zorder=3)
            self.ax.text(0.1,1.1, z, 'Layer ' + self.graphLabels[z])