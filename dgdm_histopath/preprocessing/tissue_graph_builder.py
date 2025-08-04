"""
Tissue graph construction for histopathology analysis.

Builds hierarchical graph representations of tissue from patch-level features,
incorporating spatial relationships and morphological similarities.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from typing import List, Dict, Optional, Tuple, Union
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import networkx as nx
from dataclasses import dataclass
import logging

from dgdm_histopath.preprocessing.slide_processor import SlideData, PatchInfo


@dataclass 
class GraphNode:
    """Node in tissue graph with features and metadata."""
    node_id: str
    patch_info: PatchInfo
    features: np.ndarray
    spatial_coords: Tuple[float, float]
    tissue_type: Optional[str] = None
    confidence: float = 1.0


@dataclass
class GraphEdge:
    """Edge in tissue graph with relationship features."""
    source_id: str
    target_id: str
    edge_type: str  # 'spatial', 'morphological', 'hierarchical'
    weight: float
    features: Optional[np.ndarray] = None


class TissueGraphBuilder:
    """
    Builds multi-scale tissue graphs from histopathology patches.
    
    Creates graph representations that capture both spatial adjacency
    and morphological similarity between tissue regions.
    """
    
    def __init__(
        self,
        feature_extractor: str = "dinov2",
        spatial_k: int = 8,
        morphological_k: int = 16,
        edge_threshold: float = 0.7,
        hierarchical_levels: int = 3,
        min_component_size: int = 5,
        use_adaptive_threshold: bool = True
    ):
        """
        Initialize tissue graph builder.
        
        Args:
            feature_extractor: Feature extraction method ("dinov2", "ctp", "hipt")
            spatial_k: Number of spatial neighbors for each node
            morphological_k: Number of morphological neighbors
            edge_threshold: Threshold for edge creation (0.0 to 1.0)
            hierarchical_levels: Number of hierarchical graph levels
            min_component_size: Minimum size for graph components
            use_adaptive_threshold: Whether to use adaptive thresholding
        """
        self.feature_extractor = feature_extractor
        self.spatial_k = spatial_k
        self.morphological_k = morphological_k
        self.edge_threshold = edge_threshold
        self.hierarchical_levels = hierarchical_levels
        self.min_component_size = min_component_size
        self.use_adaptive_threshold = use_adaptive_threshold
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize feature extractor
        self._init_feature_extractor()
        
    def _init_feature_extractor(self):
        """Initialize the feature extraction model."""
        if self.feature_extractor == "dinov2":
            try:
                import timm
                self.feature_model = timm.create_model(
                    'vit_base_patch14_dinov2.lvd142m',
                    pretrained=True,
                    num_classes=0  # Remove classification head
                )
                self.feature_model.eval()
                self.feature_dim = 768
            except ImportError:
                self.logger.warning("timm not available, using simple CNN features")
                self._init_simple_cnn()
        elif self.feature_extractor == "ctp":
            self._init_ctp_features()
        elif self.feature_extractor == "hipt":
            self._init_hipt_features()
        else:
            self.logger.warning(f"Unknown feature extractor: {self.feature_extractor}")
            self._init_simple_cnn()
            
    def _init_simple_cnn(self):
        """Initialize simple CNN feature extractor as fallback."""
        import torch.nn as nn
        
        self.feature_model = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512)
        )
        self.feature_dim = 512
        
    def _init_ctp_features(self):
        """Initialize CTransPath features."""
        self.logger.info("CTransPath features not implemented, using simple CNN")
        self._init_simple_cnn()
        
    def _init_hipt_features(self):
        """Initialize HIPT features.""" 
        self.logger.info("HIPT features not implemented, using simple CNN")
        self._init_simple_cnn()
        
    def extract_patch_features(self, patch: np.ndarray) -> np.ndarray:
        """
        Extract features from a single patch.
        
        Args:
            patch: RGB patch array [H, W, 3]
            
        Returns:
            Feature vector [feature_dim]
        """
        # Convert to tensor and normalize
        patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
        patch_tensor = patch_tensor.unsqueeze(0)  # Add batch dimension
        
        # Extract features
        with torch.no_grad():
            if hasattr(self.feature_model, 'forward_features'):
                features = self.feature_model.forward_features(patch_tensor)
                if len(features.shape) > 2:
                    features = features.mean(dim=[2, 3])  # Global average pooling
            else:
                features = self.feature_model(patch_tensor)
                
        return features.squeeze().numpy()
        
    def build_graph(
        self,
        slide_data: SlideData,
        patch_images: Optional[Dict[str, np.ndarray]] = None
    ) -> Data:
        """
        Build tissue graph from slide data.
        
        Args:
            slide_data: Processed slide data
            patch_images: Optional dict of patch_id -> image arrays
            
        Returns:
            PyTorch Geometric Data object
        """
        self.logger.info(f"Building graph for slide {slide_data.slide_id}")
        
        # Create graph nodes
        nodes = self._create_nodes(slide_data, patch_images)
        
        if len(nodes) == 0:
            self.logger.warning("No valid nodes created for graph")
            return self._create_empty_graph()
            
        # Create edges
        edges = self._create_edges(nodes)
        
        # Convert to PyTorch Geometric format
        graph = self._to_pytorch_geometric(nodes, edges, slide_data)
        
        self.logger.info(
            f"Created graph with {graph.num_nodes} nodes and {graph.num_edges} edges"
        )
        
        return graph
        
    def _create_nodes(
        self,
        slide_data: SlideData,
        patch_images: Optional[Dict[str, np.ndarray]] = None
    ) -> List[GraphNode]:
        """Create graph nodes from patch information."""
        nodes = []
        
        for patch in slide_data.patches:
            # Extract features if patch image is provided
            if patch_images and patch.patch_id in patch_images:
                patch_image = patch_images[patch.patch_id]
                features = self.extract_patch_features(patch_image)
            elif patch.features is not None:
                features = patch.features
            else:
                # Use placeholder features based on patch metadata
                features = self._create_placeholder_features(patch)
                
            # Normalize spatial coordinates
            spatial_coords = self._normalize_coordinates(
                patch.x, patch.y, slide_data.metadata
            )
            
            node = GraphNode(
                node_id=patch.patch_id,
                patch_info=patch,
                features=features,
                spatial_coords=spatial_coords
            )
            nodes.append(node)
            
        return nodes
        
    def _create_placeholder_features(self, patch: PatchInfo) -> np.ndarray:
        """Create placeholder features from patch metadata."""
        # Simple features based on patch properties
        features = np.array([
            patch.tissue_percentage,
            patch.magnification / 40.0,  # Normalize magnification
            float(patch.level),
            np.log1p(patch.x / 1000.0),  # Log-normalized coordinates
            np.log1p(patch.y / 1000.0),
        ])
        
        # Pad to match expected feature dimension
        if len(features) < self.feature_dim:
            padding = np.zeros(self.feature_dim - len(features))
            features = np.concatenate([features, padding])
        else:
            features = features[:self.feature_dim]
            
        return features
        
    def _normalize_coordinates(
        self, x: int, y: int, metadata: Dict
    ) -> Tuple[float, float]:
        """Normalize spatial coordinates to [0, 1] range."""
        if 'dimensions' in metadata:
            width, height = metadata['dimensions']
            norm_x = x / width
            norm_y = y / height
        else:
            # Fallback normalization
            norm_x = x / 50000.0  # Assume typical slide width
            norm_y = y / 50000.0
            
        return (norm_x, norm_y)
        
    def _create_edges(self, nodes: List[GraphNode]) -> List[GraphEdge]:
        """Create edges between graph nodes."""
        edges = []
        
        # Extract features and coordinates
        features_matrix = np.stack([node.features for node in nodes])
        coords_matrix = np.array([node.spatial_coords for node in nodes])
        
        # Create spatial edges
        spatial_edges = self._create_spatial_edges(nodes, coords_matrix)
        edges.extend(spatial_edges)
        
        # Create morphological edges
        morphological_edges = self._create_morphological_edges(nodes, features_matrix)
        edges.extend(morphological_edges)
        
        # Remove duplicate edges
        edges = self._remove_duplicate_edges(edges)
        
        return edges
        
    def _create_spatial_edges(
        self, nodes: List[GraphNode], coords: np.ndarray
    ) -> List[GraphEdge]:
        """Create edges based on spatial proximity."""
        edges = []
        
        # Use k-nearest neighbors for spatial connectivity
        nbrs = NearestNeighbors(n_neighbors=min(self.spatial_k + 1, len(nodes)))
        nbrs.fit(coords)
        
        distances, indices = nbrs.kneighbors(coords)
        
        for i, node in enumerate(nodes):
            for j in range(1, indices.shape[1]):  # Skip self (index 0)
                neighbor_idx = indices[i, j]
                distance = distances[i, j]
                
                # Create edge with distance-based weight
                weight = np.exp(-distance * 10)  # Exponential decay
                
                if weight >= self.edge_threshold:
                    edge = GraphEdge(
                        source_id=node.node_id,
                        target_id=nodes[neighbor_idx].node_id,
                        edge_type="spatial",
                        weight=weight,
                        features=np.array([distance, weight])
                    )
                    edges.append(edge)
                    
        return edges
        
    def _create_morphological_edges(
        self, nodes: List[GraphNode], features: np.ndarray
    ) -> List[GraphEdge]:
        """Create edges based on morphological similarity."""
        edges = []
        
        # Compute feature similarities
        similarities = cosine_similarity(features)
        
        # Use k-nearest neighbors based on feature similarity
        nbrs = NearestNeighbors(n_neighbors=min(self.morphological_k + 1, len(nodes)))
        nbrs.fit(features)
        
        distances, indices = nbrs.kneighbors(features)
        
        for i, node in enumerate(nodes):
            for j in range(1, indices.shape[1]):  # Skip self
                neighbor_idx = indices[i, j]
                similarity = similarities[i, neighbor_idx]
                
                if similarity >= self.edge_threshold:
                    edge = GraphEdge(
                        source_id=node.node_id,
                        target_id=nodes[neighbor_idx].node_id,
                        edge_type="morphological",
                        weight=similarity,
                        features=np.array([similarity])
                    )
                    edges.append(edge)
                    
        return edges
        
    def _remove_duplicate_edges(self, edges: List[GraphEdge]) -> List[GraphEdge]:
        """Remove duplicate edges, keeping the one with highest weight."""
        edge_dict = {}
        
        for edge in edges:
            # Create a canonical edge key (smaller id first)
            key = tuple(sorted([edge.source_id, edge.target_id]))
            
            if key not in edge_dict or edge.weight > edge_dict[key].weight:
                edge_dict[key] = edge
                
        return list(edge_dict.values())
        
    def _to_pytorch_geometric(
        self,
        nodes: List[GraphNode],
        edges: List[GraphEdge],
        slide_data: SlideData
    ) -> Data:
        """Convert graph to PyTorch Geometric Data format."""
        # Create node features matrix
        node_features = torch.tensor(
            np.stack([node.features for node in nodes]),
            dtype=torch.float
        )
        
        # Create node position matrix
        node_pos = torch.tensor(
            np.array([node.spatial_coords for node in nodes]),
            dtype=torch.float
        )
        
        # Create edge index and attributes
        if edges:
            # Map node IDs to indices
            node_id_to_idx = {node.node_id: i for i, node in enumerate(nodes)}
            
            edge_index = []
            edge_attr = []
            edge_types = []
            
            for edge in edges:
                src_idx = node_id_to_idx[edge.source_id]
                tgt_idx = node_id_to_idx[edge.target_id]
                
                # Add both directions for undirected graph
                edge_index.extend([[src_idx, tgt_idx], [tgt_idx, src_idx]])
                
                # Edge attributes
                if edge.features is not None:
                    edge_attr.extend([edge.features, edge.features])
                else:
                    edge_attr.extend([np.array([edge.weight]), np.array([edge.weight])])
                    
                # Edge types
                type_encoding = {"spatial": 0, "morphological": 1, "hierarchical": 2}
                edge_type = type_encoding.get(edge.edge_type, 0)
                edge_types.extend([edge_type, edge_type])
                
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_attr = torch.tensor(np.stack(edge_attr), dtype=torch.float)
            edge_types = torch.tensor(edge_types, dtype=torch.long)
        else:
            # Empty graph
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)
            edge_types = torch.empty((0,), dtype=torch.long)
            
        # Create PyTorch Geometric Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=node_pos,
            edge_type=edge_types,
            num_nodes=len(nodes)
        )
        
        # Add metadata
        data.slide_id = slide_data.slide_id
        data.num_patches = len(slide_data.patches)
        data.slide_metadata = slide_data.metadata
        
        return data
        
    def _create_empty_graph(self) -> Data:
        """Create empty graph for cases with no valid nodes."""
        return Data(
            x=torch.empty((0, self.feature_dim), dtype=torch.float),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 1), dtype=torch.float),
            pos=torch.empty((0, 2), dtype=torch.float),
            num_nodes=0
        )
        
    def create_hierarchical_graph(
        self, base_graph: Data, levels: int = 3
    ) -> List[Data]:
        """
        Create hierarchical representation with multiple resolution levels.
        
        Args:
            base_graph: Base graph at highest resolution
            levels: Number of hierarchical levels
            
        Returns:
            List of graphs at different resolution levels
        """
        graphs = [base_graph]
        current_graph = base_graph
        
        for level in range(1, levels):
            # Pool nodes to create coarser graph
            pooled_graph = self._pool_graph(current_graph, pool_ratio=0.5)
            graphs.append(pooled_graph)
            current_graph = pooled_graph
            
        return graphs
        
    def _pool_graph(self, graph: Data, pool_ratio: float = 0.5) -> Data:
        """Pool graph nodes to create coarser representation."""
        num_nodes = graph.num_nodes
        num_keep = max(1, int(num_nodes * pool_ratio))
        
        if num_keep >= num_nodes:
            return graph
            
        # Simple pooling based on node degree
        degrees = torch.zeros(num_nodes)
        if graph.edge_index.numel() > 0:
            for i in range(num_nodes):
                degrees[i] = (graph.edge_index[0] == i).sum() + (graph.edge_index[1] == i).sum()
                
        # Keep nodes with highest degrees
        _, keep_indices = torch.topk(degrees, num_keep)
        keep_indices = keep_indices.sort()[0]
        
        # Create new graph with selected nodes
        new_x = graph.x[keep_indices]
        new_pos = graph.pos[keep_indices]
        
        # Update edge index
        node_map = torch.full((num_nodes,), -1, dtype=torch.long)
        node_map[keep_indices] = torch.arange(num_keep)
        
        if graph.edge_index.numel() > 0:
            edge_mask = (node_map[graph.edge_index[0]] >= 0) & (node_map[graph.edge_index[1]] >= 0)
            new_edge_index = graph.edge_index[:, edge_mask]
            new_edge_index = node_map[new_edge_index]
            
            new_edge_attr = graph.edge_attr[edge_mask] if graph.edge_attr.numel() > 0 else torch.empty((0, 1))
        else:
            new_edge_index = torch.empty((2, 0), dtype=torch.long)
            new_edge_attr = torch.empty((0, 1), dtype=torch.float)
            
        pooled_graph = Data(
            x=new_x,
            edge_index=new_edge_index,
            edge_attr=new_edge_attr,
            pos=new_pos,
            num_nodes=num_keep
        )
        
        return pooled_graph