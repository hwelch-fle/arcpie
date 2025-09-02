from arcpy import (
    Polyline, 
    PointGeometry,
)
from featureclass import FeatureClass

import networkx as nx
from collections.abc import (
    Sequence, 
    Iterator,
)
from typing import Any

class FeatureGraph:
    def __init__(self, edges: FeatureClass[Polyline], nodes: FeatureClass[PointGeometry], tolerance: float=0.0,
                 *,
                 node_attributes: Sequence[str] | None=None,
                 edge_attributes: Sequence[str] | None=None) -> None:
        self.node_features = nodes
        self.edge_features = edges
        self.node_attributes = node_attributes or tuple()
        self.edge_attributes = edge_attributes or tuple()
        self.tolerance = tolerance
        self._graph = self.build_graph()
        self.user_nodes: list[tuple[int, PointGeometry]] = []

    @property
    def graph(self) -> nx.Graph:
        return self._graph
    
    @property
    def nodes(self) -> Iterator[tuple[int, dict[str, Any]]]:
        return self.graph.nodes.data()

    def refresh(self) -> None:
        self.user_nodes = []
        self._graph = self.build_graph()

    def build_graph(self) -> nx.Graph:
        """Build a graph from the provided features
        Structure of the Graph will be:
        
        `node:oid[attrs] <-(edge:[attrs])-> node:oid[attrs]`
        
        e.g.
        
        `45: {'name': 'bill'} <-{'length': 1200.4, 'link_name': 'direct'}-> 46: {'name': 'sue'}`
        
        Returns:
            (nx.Graph): A networkx Graph with all points connected by the provided edges
        
        """

        # Initialize an undirected graph
        g = nx.Graph()

        # Add all points as nodes (with specified attributes)
        for oid, node, *node_attrs in self.node_features[('OID@', 'SHAPE@', *self.node_attributes)]:
            user_attrs: dict[str, Any] = dict(zip(self.node_attributes, node_attrs))
            system_attrs: dict[str, Any] = {'OID@': oid, 'SHAPE@': node}
            # Merge user and system attrs then unpack into **attr
            g.add_node(oid, **{**user_attrs, **system_attrs})

        # Connect all nodes using edges (with specified attributes)
        for oid, edge, *edge_attrs in self.edge_features[('OID@', 'SHAPE@', *self.edge_attributes)]:
            edge: Polyline
            fp = PointGeometry(edge.firstPoint)
            lp = PointGeometry(edge.lastPoint)

            # Buffer anything with a specified non-zero tolerance
            if self.tolerance:
                fp = fp.buffer(self.tolerance)
                lp = lp.buffer(self.tolerance)

            # Get all nodes that Intersect the endpoints of the edge
            with self.node_features.spatial_filter(fp.union(lp)):
                to_add: list[int] = list(self.node_features['OID@'])
            
            # Generate all unique connections for the edge and add them to the graph with the edge attrs
            # avoid connecting nodes to themselves
            for cxn in {tuple(sorted([a, b])) for a in to_add for b in to_add if a != b}:
                user_attrs: dict[str, Any] = dict(zip(self.edge_attributes, edge_attrs))
                system_attrs: dict[str, Any] = {'length': edge.length, 'OID@': oid, 'SHAPE@': edge}
                # Merge user and system attrs then unpack into **attr
                g.add_edge(cxn[0], cxn[1], **{**user_attrs, **system_attrs})
        return g        

    @overload
    def __contains__(self, node: int) -> bool: ...
    @overload
    def __contains__(self, node: PointGeometry) -> bool: ...

    def __contains__(self, node: int | PointGeometry) -> bool:
        if not isinstance(node, PointGeometry):
            return node in self.graph
        for _, data in self.nodes:
            shape: PointGeometry = data['SHAPE@']
            if shape == node:
                return True
        return False

    def index_of(self, node: PointGeometry) -> int:
        for oid, data in self.nodes:
            oid: int
            shape: PointGeometry = data['SHAPE@']
            if node == shape:
                return oid
            
        raise IndexError(f'Node: {node} not found in graph!')

    def shortest_path(self, from_node: int|PointGeometry, to_node: int|PointGeometry) -> Iterator[Polyline]:
        """Return the line geometries that make up the shortest path between the provided nodes
        
        Args:
            from_node (int|PointGeometry): The starting node in the graph
            to_node (int|PointGeometry): The ending node in the graph

        Returns:
            (Iterator[Polyline]): An iterator of Polyline geometries that make up the shortest path 
        """
        if isinstance(from_node, PointGeometry):
            from_node = self.index_of(from_node)

        if isinstance(to_node, PointGeometry):
            to_node = self.index_of(to_node)

        route: list[int] = nx.shortest_path(self.graph, from_node, to_node, weight='length')

        for i in range(len(route)-1):
            yield self.graph.get_edge_data(route[i], route[i+1])['SHAPE@']
        
    def add_node(self, node: PointGeometry, **data: Any) -> int:
        """Adds a node to the graph
        
        Note:
            User added nodes will be assigned negative ids to prevent collision with system nodes from the base featureclass.
            These nodes will also be removed if the graph is refreshed, this method is primarily for checking a geometry against a 
            network as a one off.
        
        Args:
            node (PointGeometry): The node to add
            **data (Any): User defined attributes to add to the node

        Returns:
            (int): The new index of the node (negative indexed)
        """
        new_index = -1*(len(self.user_nodes)+1)
        user_attrs = data
        system_attrs: dict[str, Any] = {'OID@': new_index, 'SHAPE@': node}
        self.graph.add_node(new_index, **{**user_attrs, **system_attrs})
        self.user_nodes.append((new_index, node))

        # Buffer anything with a specified non-zero tolerance
        if self.tolerance:
            node_buff = node.buffer(self.tolerance)
        else:
            node_buff = node
        
        with self.edge_features.spatial_filter(node_buff):
            connecting_edges = tuple(self.edge_features[('OID@', 'SHAPE@', *self.edge_attributes)])
            
        for oid, edge_shape, *edge_attrs in connecting_edges:
            oid: int
            edge_shape: Polyline
            
            fp = PointGeometry(edge_shape.firstPoint)
            lp = PointGeometry(edge_shape.lastPoint)

            # Buffer anything with a specified non-zero tolerance
            if self.tolerance:
                fp = fp.buffer(self.tolerance)
                lp = lp.buffer(self.tolerance)
            
            with self.node_features.spatial_filter(fp.union(lp)):
                cxns = list(self.node_features['OID@'])
            
            user_attrs = dict(zip(self.edge_attributes, edge_attrs))
            system_attrs: dict[str, Any] = {'length': edge_shape.length, 'OID@': oid, 'SHAPE@': edge_shape}
            self.graph.add_edges_from([(new_index, n) for n in cxns], **{**user_attrs, **system_attrs})
        return new_index