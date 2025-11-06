import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_add, scatter_max, scatter_mean
from data_utils import *
from mpqeutils import *
from typing import Dict, Any

from base import BaseGNN
from mpqeutils import margin_loss


# Import loss functions from utils

class RGCNEncoderDecoder(BaseGNN):
    """
    RGCN Encoder-Decoder that inherits from BaseGNN with YAML configuration support.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RGCN Encoder-Decoder.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        # Initialize without graph and enc - they will be set later
        self.graph = None
        self.enc = None
        
        # Set embedding dimension from config
        self.emb_dim = config.get('embed_dim', 128)
        
        # Set BaseGNN required parameters
        config['input_dim'] = self.emb_dim
        config['hidden_dim'] = self.emb_dim  
        config['output_dim'] = self.emb_dim
        
        # Call parent constructor FIRST
        super(RGCNEncoderDecoder, self).__init__(config)
        
        # Loss function configuration
        self.loss_function_name = config.get('loss_function', 'margin_loss')
        self.margin = config.get('margin', 1.0)
        self.temperature = config.get('temperature', 0.1)
        
        # Extract RGCN-specific parameters using config.get() with defaults
        self.readout_str = config.get('readout', 'mp')
        self.scatter_op = config.get('scatter_op', 'add')
        self.shared_layers = config.get('shared_layers', True)
        self.adaptive = config.get('adaptive', True)
        
        # Training parameters
        self.max_burn_in = config.get('max_burn_in', 100000)
        self.batch_size = config.get('batch_size', 512)
        self.log_every = config.get('log_every', 500)
        self.val_every = config.get('val_every', 1000)
        self.tol = config.get('tol', 1e-6)
        self.inter_weight = config.get('inter_weight', 0.005)
        self.path_weight = config.get('path_weight', 0.01)
        
        # Training state variables
        self.edge_conv = False
        self.ema_loss = None
        self.vals = []
        self.losses = []
        self.conv_test = None
        self.train_iterators = {}
        self.current_iter = 0

    def _get_loss_function(self):
        """Get the specified loss function from utils."""
        loss_functions = {
            'margin_loss': margin_loss
        }
        
        if self.loss_function_name not in loss_functions:
            raise ValueError(f"Unknown loss function: {self.loss_function_name}. "
                           f"Available options: {list(loss_functions.keys())}")
        
        return loss_functions[self.loss_function_name]
    
    def set_graph_and_encoder(self, graph, enc):
        """
        Set the graph and encoder objects and initialize RGCN components.
        
        Args:
            graph: Graph object with relations and mode_weights
            enc: Encoder object for node embeddings
        """
        self.graph = graph
        self.enc = enc
        
        # Update embedding dimension from graph if available
        if hasattr(graph, 'feature_dims') and graph.feature_dims:
            self.emb_dim = graph.feature_dims[next(iter(graph.feature_dims))]
        
        # Now setup RGCN components that depend on graph
        self._setup_rgcn_components()
        
        # Initialize optimizer now that we have parameters
        self._init_optimizer_and_criterion()
    
    def _setup_rgcn_components(self):
        """Set up RGCN-specific components."""
        if self.graph is None:
            raise ValueError("Graph must be set before setting up RGCN components")
            
        # Initialize mode embeddings
        self.mode_embeddings = nn.Embedding(len(self.graph.mode_weights), self.emb_dim)
        
        # Create mode and relation ID mappings
        self.mode_ids = {}
        mode_id = 0
        for mode in self.graph.mode_weights:
            self.mode_ids[mode] = mode_id
            mode_id += 1

        self.rel_ids = {}
        id_rel = 0
        for r1 in self.graph.relations:
            for r2 in self.graph.relations[r1]:
                rel = (r1, r2[1], r2[0])
                self.rel_ids[rel] = id_rel
                id_rel += 1

        # Initialize RGCN layers
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            if len(self.layers) == 0 or not self.shared_layers:
                rgcn = RGCNConv(in_channels=self.emb_dim,
                                out_channels=self.emb_dim,
                                num_relations=len(self.graph.rel_edges),
                                num_bases=0)
            self.layers.append(rgcn)

        # Set up scatter function
        if self.scatter_op == 'add':
            scatter_fn = scatter_add
        elif self.scatter_op == 'max':
            scatter_fn = scatter_max
        elif self.scatter_op == 'mean':
            scatter_fn = scatter_mean
        else:
            raise ValueError(f'Unknown scatter op {self.scatter_op}')

        # Initialize readout function
        if self.readout_str == 'sum':
            self.readout = self.sum_readout
        elif self.readout_str == 'max':
            self.readout = self.max_readout
        elif self.readout_str == 'mlp':
            self.readout = MLPReadout(self.emb_dim, self.emb_dim, scatter_fn)
        elif self.readout_str == 'targetmlp':
            self.readout = TargetMLPReadout(self.emb_dim, scatter_fn)
        elif self.readout_str == 'concat':
            self.readout = MLPReadout(self.emb_dim * self.num_layers, self.emb_dim, scatter_fn)
        elif self.readout_str == 'mp':
            self.readout = self.target_message_readout
        else:
            raise ValueError(f'Unknown readout function {self.readout_str}')
    
    def _init_optimizer_and_criterion(self):
        """Initialize optimizer and criterion - override for RGCN-specific needs."""
        # Only initialize optimizer if we have parameters (i.e., after layers are created)
        if self.optimizer is None and list(self.parameters()):
            self.optimizer = self._create_optimizer()
        
        if self.criterion is None:
            # RGCN uses custom loss, but set a default for BaseGNN compatibility
            self.criterion = nn.MSELoss()  
    
    def compute_loss(self, formula, queries, anchor_ids=None, var_ids=None,
                    q_graphs=None, hard_negatives=False):
        """
        Compute loss using the configured loss function.
        
        Args:
            formula: Query formula
            queries: List of query objects
            anchor_ids: Anchor node IDs (optional)
            var_ids: Variable node IDs (optional)
            q_graphs: Query graphs (optional)
            hard_negatives: Whether to use hard negatives
            
        Returns:
            Computed loss value
        """
        loss_function = self._get_loss_function()
        
        # Pass additional parameters based on loss function
        loss_kwargs = {
            'margin': self.margin,
            # 'temperature': self.temperature
        }
        
        return loss_function(
            self, formula, queries, anchor_ids, var_ids, q_graphs, 
            hard_negatives, **loss_kwargs
        )
    
    def forward(self, first_arg, second_arg=None, third_arg=None, **kwargs):
        """
        Smart dispatcher forward method that handles both GNN tensor operations 
        and query processing based on argument types.
        
        Two modes:
        1. GNN mode: forward(x, edge_index, **kwargs) - for tensor-based operations
        2. Query mode: forward(formula, queries, target_nodes, **kwargs) - for query processing
        """
        # Smart dispatching based on first argument type
        if isinstance(first_arg, torch.Tensor):
            # GNN mode: forward(x, edge_index, **kwargs)
            return self._tensor_forward(first_arg, second_arg, **kwargs)
        else:
            # Query mode: forward(formula, queries, target_nodes, **kwargs)
            return self.forward_query(first_arg, second_arg, third_arg, **kwargs)

    def _tensor_forward(self, x, edge_index, **kwargs):
        """
        Standard forward pass for tensor-based GNN operations.
        This is the original forward logic moved to a separate method.
        """
        if self.layers is None or len(self.layers) == 0:
            raise ValueError("RGCN layers not initialized. Call set_graph_and_encoder() first.")
            
        # Basic RGCN forward pass
        h = x
        for i in range(self.num_layers - 1):
            h = self.layers[i](h, edge_index, kwargs.get('edge_type'))
            h = F.relu(h)
        
        h = self.layers[-1](h, edge_index, kwargs.get('edge_type'))
        return h

    def forward_query(self, formula, queries, target_nodes,
                      anchor_ids=None, var_ids=None, q_graphs=None,
                      neg_nodes=None, neg_lengths=None):
        """
        Original RGCN forward method for query processing.
        """
        if self.graph is None or self.enc is None:
            raise ValueError("Graph and encoder must be set before running queries. Call set_graph_and_encoder() first.")
            
        if anchor_ids is None or var_ids is None or q_graphs is None:
            from data_utils import RGCNQueryDataset
            query_data = RGCNQueryDataset.get_query_graph(formula, queries,
                                                          self.rel_ids,
                                                          self.mode_ids)
            anchor_ids, var_ids, q_graphs = query_data

        device = next(self.parameters()).device
        var_ids = var_ids.to(device)
        q_graphs = q_graphs.to(device)

        batch_size, num_anchors = anchor_ids.shape
        n_vars = var_ids.shape[0]
        n_nodes = num_anchors + n_vars

        x = torch.empty(batch_size, n_nodes, self.emb_dim).to(var_ids.device)
        for i, anchor_mode in enumerate(formula.anchor_modes):
            x[:, i] = self.enc(anchor_ids[:, i], anchor_mode).t()
        x[:, num_anchors:] = self.mode_embeddings(var_ids)
        x = x.reshape(-1, self.emb_dim)
        q_graphs.x = x

        if self.adaptive:
            from data_utils import RGCNQueryDataset
            num_passes = RGCNQueryDataset.query_diameters[formula.query_type]
            if num_passes > len(self.layers):
                raise ValueError(f'RGCN is adaptive with {len(self.layers)}'
                                 f' layers, but query requires {num_passes}.')
        else:
            num_passes = self.num_layers

        h1 = q_graphs.x
        h_layers = []
        for i in range(num_passes - 1):
            h1 = self.layers[i](h1, q_graphs.edge_index, q_graphs.edge_type)
            h1 = F.relu(h1)
            if self.readout_str == 'concat':
                h_layers.append(h1)

        h1 = self.layers[-1](h1, q_graphs.edge_index, q_graphs.edge_type)

        if self.readout_str == 'concat':
            h_layers.append(h1)
            h1 = torch.cat(h_layers, dim=1)

        out = self.readout(embs=h1, batch_idx=q_graphs.batch,
                           batch_size=batch_size, num_nodes=n_nodes,
                           num_anchors=num_anchors)

        target_embeds = self.enc(target_nodes, formula.target_mode).t()
        scores = F.cosine_similarity(out, target_embeds, dim=1)

        if neg_nodes is not None:
            neg_embeds = self.enc(neg_nodes, formula.target_mode).t()
            out = out.repeat_interleave(torch.tensor(neg_lengths).to(device),
                                        dim=0)
            neg_scores = F.cosine_similarity(out, neg_embeds)
            scores = torch.cat((scores, neg_scores), dim=0)

        return scores

    # Keep all the existing readout methods
    def sum_readout(self, embs, batch_idx, **kwargs):
        return scatter_add(embs, batch_idx, dim=0)

    def max_readout(self, embs, batch_idx, **kwargs):
        out, argmax = scatter_max(embs, batch_idx, dim=0)
        return out

    def target_message_readout(self, embs, batch_size, num_nodes, num_anchors, **kwargs):
        device = embs.device
        non_target_idx = torch.ones(num_nodes, dtype=torch.bool)
        non_target_idx[num_anchors] = 0
        non_target_idx.to(device)

        embs = embs.reshape(batch_size, num_nodes, -1)
        targets = embs[:, ~non_target_idx].reshape(batch_size, -1)
        return targets

    def _run_batch_with_new_loss(self, queries_iterator, hard_negatives=False):
        """Run a batch using the new loss computation method."""
        self.train()
        batch = next(queries_iterator)
        return self.compute_loss(*batch, hard_negatives=hard_negatives)

    def train_step(self, data, mask=None):
        """
        Single training step for RGCN query reasoning.

        Args:
            data: Should contain train_queries, current_iteration, and other training state
            mask: Not used for RGCN, kept for interface compatibility

        Returns:
            Average loss for this step
        """
        if self.graph is None or self.enc is None:
            raise ValueError("Graph and encoder must be set before training. "
                            "Call set_graph_and_encoder() first.")
            
        self.train()

        # Extract training data
        train_queries = data.train_queries
        batch_size = getattr(data, 'batch_size', self.batch_size)

        # Always rebuild iterators fresh for this step
        self.train_iterators = {}
        for query_type, queries in train_queries.items():
            if queries:  # only build if queries exist for this type
                self.train_iterators[query_type] = get_queries_iterator(queries, batch_size, self)

        self.optimizer.zero_grad()

        # Always train on 1-chain if available
        if "1-chain" not in self.train_iterators:
            raise ValueError("No 1-chain queries found — training requires at least 1-chain queries.")
        loss = self._run_batch_with_new_loss(self.train_iterators["1-chain"])

        # If past burn-in, add other query types
        if getattr(data, "past_burn_in", False):
            for query_type in train_queries:
                if query_type == "1-chain":
                    continue
                if query_type not in self.train_iterators:
                    continue  # skip missing query types safely

                if "inter" in query_type:
                    loss += self.inter_weight * self._run_batch_with_new_loss(self.train_iterators[query_type])
                    loss += self.inter_weight * self._run_batch_with_new_loss(
                        self.train_iterators[query_type], hard_negatives=True
                    )
                else:
                    loss += self.path_weight * self._run_batch_with_new_loss(self.train_iterators[query_type])

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def show_detailed_predictions(self, type_queries, query_type, max_queries=3):
        """Show input queries and model predictions"""
        
        query_count = 0
        for formula, formula_queries in type_queries.items():
            if query_count >= max_queries:
                break
                
            for i, query in enumerate(formula_queries[:2]):
                if query_count >= max_queries:
                    break
                    
                try:
                    # Prepare the correct arguments for forward_query
                    queries = [query]  # Put query in a list
                    target_nodes = [query.target_node]  # Target nodes as list
                    
                    # Try the forward_query method with correct parameters
                    scores = self.forward_query(queries, target_nodes)
                    
                    # Prepare input query format
                    anchor_nodes = query.anchor_nodes
                    target_node = query.target_node
                    neg_samples = query.neg_samples
                    
                    if isinstance(anchor_nodes, tuple) and len(anchor_nodes) > 1:
                        input_query = f"({query_type}: ({', '.join(map(str, anchor_nodes))}) -> ?)"
                    else:
                        anchor_str = anchor_nodes[0] if isinstance(anchor_nodes, (list, tuple)) else anchor_nodes
                        input_query = f"({query_type}: {anchor_str} -> ?)"
                    
                    # Handle the scores output
                    if isinstance(scores, torch.Tensor):
                        scores_np = scores.cpu().numpy()
                        
                        # Handle different score shapes
                        if len(scores_np.shape) > 1:
                            scores_np = scores_np.flatten()
                        
                        if len(scores_np) > 1:  # Multiple predictions
                            top_indices = np.argsort(scores_np)[-5:][::-1]
                            top_scores = scores_np[top_indices]
                            predictions = [(int(idx), float(score)) for idx, score in zip(top_indices, top_scores)]
                            
                            target_score = float(scores_np[target_node]) if target_node < len(scores_np) else 0.0
                            neg_score = float(scores_np[neg_samples[0]]) if neg_samples and neg_samples[0] < len(scores_np) else 0.0
                        else:  # Single prediction score
                            predictions = [(target_node, float(scores_np[0]))]
                            target_score = float(scores_np[0])
                            neg_score = 0.0
                    else:
                        predictions = [(target_node, float(scores))]
                        target_score = float(scores)
                        neg_score = 0.0
                    
                    # Print results
                    print(f"Input: {input_query}")
                    print(f"Output: {predictions}")
                    print(f"Target {target_node}: {target_score:.6f}, Negative {neg_samples[0] if neg_samples else 'N/A'}: {neg_score:.6f}")
                    print()
                    
                    query_count += 1
                    
                except Exception as e:
                    print(f"Error: {e}")
                    query_count += 1

    def eval_step(self, data, mask=None):
        """Single evaluation step for RGCN query reasoning."""
        if self.graph is None or self.enc is None:
            raise ValueError("Graph and encoder must be set before evaluation. Call set_graph_and_encoder() first.")
            
        self.eval()
        
        queries = getattr(data, 'val_queries', getattr(data, 'test_queries', None))
        iteration = getattr(data, 'current_iteration', 0)
        batch_size = getattr(data, 'batch_size', 128)
        
        if queries is None:
            print("ERROR: No queries found in data")
            return {'loss': float('inf'), 'accuracy': 0.0}
        
        vals = {}
        total_loss = 0.0
        num_query_types = 0
        
        with torch.no_grad():
            for query_type in queries["one_neg"]:
                try:
                    auc, rel_aucs = eval_auc_queries(queries["one_neg"][query_type], self)
                    perc = eval_perc_queries(queries["full_neg"][query_type], self, batch_size)
                    vals[query_type] = auc
                    total_loss += (1.0 - auc)
                    num_query_types += 1
                    
                    print(f"{query_type} val AUC: {auc:.6f} val perc {perc}; iteration: {iteration}")
                    
                    if "inter" in query_type:
                        auc_hard, _ = eval_auc_queries(queries["one_neg"][query_type], self, hard_negatives=True)
                        vals[query_type + "hard"] = auc_hard
                        total_loss += (1.0 - auc_hard)
                        num_query_types += 1
                        print(f"Hard-{query_type} val AUC: {auc_hard:.6f}; iteration: {iteration}")
                        
                except Exception as e:
                    print(f"ERROR in eval_step for {query_type}: {e}")
                    import traceback
                    traceback.print_exc()
                    vals[query_type] = 0.0
                    total_loss += 1.0
                    num_query_types += 1
        
        avg_loss = total_loss / max(num_query_types, 1)
        avg_auc = np.mean(list(vals.values())) if vals else 0.0
        
        print(f"Final evaluation - avg_auc: {avg_auc}, vals: {vals}")
        
        return {'loss': avg_loss, 'accuracy': avg_auc}
    
    def infer_step(self, queries, top_k=5, show_details=True):
        """
        Perform inference on a list of queries and return top-k results.
        
        Args:
            queries: List of Query objects or dict of {query_type: {formula: [queries]}}
            top_k: Number of top results to return for each query
            show_details: Whether to print detailed input/output information
            
        Returns:
            Dictionary containing inference results
        """
        if self.graph is None or self.enc is None:
            raise ValueError("Graph and encoder must be set before inference. Call set_graph_and_encoder() first.")
        
        self.eval()
        results = {}
        
        # Handle different input formats
        if isinstance(queries, list):
            # Convert list to dict format
            query_dict = defaultdict(lambda: defaultdict(list))
            for query in queries:
                query_dict[query.formula.query_type][query.formula].append(query)
            queries = query_dict
        
        with torch.no_grad():
            for query_type in queries:
                if show_details:
                    print(f"\n{'='*60}")
                    print(f"PROCESSING QUERY TYPE: {query_type}")
                    print(f"{'='*60}")
                
                results[query_type] = {}
                
                for formula in queries[query_type]:
                    formula_queries = queries[query_type][formula]
                    if show_details:
                        print(f"\nFormula: {formula}")
                        print(f"Number of queries: {len(formula_queries)}")
                    
                    formula_results = []
                    
                    # Process queries in batches
                    batch_size = 32
                    for i in range(0, len(formula_queries), batch_size):
                        batch_queries = formula_queries[i:i+batch_size]
                        
                        for query in batch_queries:
                            query_result = self._process_single_query(query, formula, top_k, show_details)
                            formula_results.append(query_result)
                    
                    results[query_type][formula] = formula_results
        
        return results

    def _process_single_query(self, query, formula, top_k, show_details):
        """
        Process a single query and return results.
        """
        # Get all possible target nodes for this mode
        target_mode = formula.target_mode
        all_targets = list(self.graph.full_lists[target_mode])
        
        if show_details:
            print(f"\n--- Single Query Processing ---")
            print(f"Query type: {query.formula.query_type}")
            print(f"Anchor nodes: {query.anchor_nodes}")
            print(f"Target mode: {target_mode}")
            print(f"True target: {query.target_node}")
            
            # Show query structure based on type
            if hasattr(query, 'query_graph') and query.query_graph:
                print(f"Query structure: {query.query_graph}")
        
        # Score the true target ONLY if it's a valid node (not -1)
        true_score = None
        if query.target_node != -1:
            true_target_score = self.forward_query(formula, [query], [query.target_node])
            true_score = float(true_target_score.cpu().numpy()[0])
            
            if show_details:
                print(f"True target score: {true_score:.6f}")
        else:
            if show_details:
                print(f"True target unknown (placeholder: -1) - predicting from all candidates")
        
        # Handle different negative sampling strategies
        if hasattr(query, 'neg_samples') and query.neg_samples:
            # Use provided negative samples
            candidates = list(query.neg_samples[:min(1000, len(query.neg_samples))])
            # Only add true target if it's valid
            if query.target_node != -1:
                candidates = [query.target_node] + candidates
        else:
            # Use all possible targets as candidates
            candidates = all_targets
        
        # Score all candidates
        candidate_scores = self.forward_query(formula, [query] * len(candidates), candidates)
        scores_np = candidate_scores.cpu().numpy()
        
        # Create results with node IDs and scores
        scored_candidates = [(candidates[i], float(scores_np[i])) for i in range(len(candidates))]
        
        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Find rank of true target (only if it's valid)
        true_target_rank = None
        if query.target_node != -1:
            for rank, (node_id, score) in enumerate(scored_candidates):
                if node_id == query.target_node:
                    true_target_rank = rank + 1
                    break
        
        # Get top-k results
        top_results = scored_candidates[:top_k]
        
        if show_details:
            print(f"\nTop-{top_k} predictions:")
            for rank, (node_id, score) in enumerate(top_results):
                marker = " ✓" if node_id == query.target_node else ""
                print(f"  Rank {rank+1}: Node {node_id} (score: {score:.6f}){marker}")
            
            if true_target_rank:
                print(f"True target rank: {true_target_rank}")
            
            # Show some negative samples if available
            if hasattr(query, 'neg_samples') and query.neg_samples:
                neg_sample = query.neg_samples[0]
                neg_idx = candidates.index(neg_sample) if neg_sample in candidates else None
                if neg_idx:
                    neg_score = float(scores_np[neg_idx])
                    print(f"First negative sample {neg_sample}: {neg_score:.6f}")
        
        return {
            'query': query,
            'true_target': query.target_node if query.target_node != -1 else None,
            'true_score': true_score,
            'true_rank': true_target_rank,
            'top_predictions': top_results,
            'total_candidates': len(candidates)
        }

    def get_model_info(self):
        """Override to include RGCN-specific information."""
        info = super().get_model_info()
        info.update({
            'emb_dim': self.emb_dim,
            'readout_function': self.readout_str,
            'scatter_operation': self.scatter_op,
            'adaptive': self.adaptive,
            'shared_layers': self.shared_layers,
            'num_relations': len(self.graph.rel_edges) if (self.graph and hasattr(self.graph, 'rel_edges')) else 'unknown',
            'loss_function': self.loss_function_name
        })
        return info