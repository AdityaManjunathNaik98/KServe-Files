from ts.torch_handler.base_handler import BaseHandler
from mpqeutils import JSONQueryBuilder
import pickle
import torch
import json
import os


class MPQEHandler(BaseHandler):
    def initialize(self, context):
        # DON'T call super().initialize() - do custom loading
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create config (based on your YAML files)
        config = {
            'model_name': 'mpqe',
            'embed_dim': 128,
            'depth': 0,
            'use_cuda': torch.cuda.is_available(),
            'data_dir': model_dir
        }
        
        # Import and create model
        from registry import create_model
        self.model = create_model('mpqe', config)
        
        # Load graph and encoder
        from data_utils import load_graph
        from mpqeutils import get_encoder, cudify
        
        graph, feature_modules, node_maps = load_graph(model_dir, embed_dim=128)
        
        if config['use_cuda']:
            features = cudify(feature_modules, node_maps)
            for key in node_maps:
                node_maps[key] = node_maps[key].cuda()
        
        out_dims = {mode: config['embed_dim'] for mode in graph.relations}
        enc = get_encoder(
            depth=config['depth'], 
            graph=graph, 
            out_dims=out_dims,
            feature_modules=feature_modules, 
            cuda=config['use_cuda']
        )
        
        # Set graph and encoder in model
        self.model.set_graph_and_encoder(graph, enc)

        self.query_builder = JSONQueryBuilder(graph, self.model.rel_ids, self.model.mode_ids)

        # Load the trained weights
        model_path = os.path.join(model_dir, "mpqe.pth")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
    
    # def preprocess(self, requests):
    #     all_queries = []
    #     for req in requests:
    #         # Get the nested data structure
    #         if 'body' in req and 'data' in req['body']:
    #             data = req['body']['data']  # This is the JSON string
    #         else:
    #             data = req.get("data") or req.get("body") or req
            
    #         if data:
    #             # Parse the JSON string
    #             if isinstance(data, str):
    #                 input_data = json.loads(data)
    #             else:
    #                 input_data = data
                
    #             # Extract queries
    #             if 'queries' in input_data:
    #                 all_queries.extend(input_data['queries'])
        
    #     print(f"Preprocessed {len(all_queries)} queries")
    #     return all_queries
    def preprocess(self, requests):
        all_queries = []
        for req in requests:
            if 'body' in req and 'data' in req['body']:
                data = req['body']['data']
            else:
                data = req.get("data") or req.get("body") or req
            
            if data:
                input_data = json.loads(data) if isinstance(data, str) else data
                
                # Handle the tuple format directly
                if 'query_tuple' in input_data:
                    all_queries.extend(input_data['query_tuple'])
                elif 'queries' in input_data:
                    all_queries.extend(input_data['queries'])
        
        print(f"Preprocessed {len(all_queries)} queries")
        return all_queries

    def inference(self, json_queries):
        print(f"Starting inference with {len(json_queries)} queries")  # Add this
        
        queries_to_process = []
        for json_query in json_queries:
            try:
                query = self.query_builder.build_query_from_json(json_query, sample_negatives=True)
                if query is not None:
                    queries_to_process.append(query)
            except Exception as e:
                print(f"Failed to build query: {e}")  # Add this
                continue
        
        print(f"Built {len(queries_to_process)} Query objects")  # Add this
        
        if not queries_to_process:
            return {}
        
        results = self.model.infer_step(queries_to_process, top_k=5)
        print(f"Inference results: {results}")  # Add this
        return results
    
    def postprocess(self, results):
        print(f"Postprocessing results: {results}")  # Debug log
        
        if not results:
            return ["[]"]
        
        formatted_results = []
        
        # Extract results from the nested structure
        for query_type in results:
            for formula in results[query_type]:
                for query_result in results[query_type][formula]:
                    result = {
                        'query_type': query_type,
                        'anchor_nodes': list(query_result['query'].anchor_nodes),
                        'target_mode': str(query_result['query'].formula.target_mode),
                        'top_predictions': [
                            {'rank': i + 1, 'node_id': int(node_id), 'score': float(score)}
                            for i, (node_id, score) in enumerate(query_result['top_predictions'])
                        ],
                        'total_candidates': query_result['total_candidates']
                    }
                    formatted_results.append(result)
        
        print(f"Formatted results: {formatted_results}")  # Debug log
        return [json.dumps(formatted_results, indent=2)]