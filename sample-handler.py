from ts.torch_handler.base_handler import BaseHandler
import torch
import pickle
import json

class MPQEHandler(BaseHandler):
    def initialize(self, context):
        super().initialize(context)
        
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        
        # Load graph and encoder
        from nesy_factory.utils.data_utils import load_graph
        from nesy_factory.utils.mpqeutils import get_encoder, cudify
        
        graph, feature_modules, node_maps = load_graph(model_dir, embed_dim=128)
        features = cudify(feature_modules, node_maps)
        enc = get_encoder(depth=0, graph=graph, out_dims=graph.feature_dims, 
                         feature_modules=feature_modules, cuda=False)
        
        # Set graph and encoder in the loaded model
        self.model.set_graph_and_encoder(graph, enc)
    
    def preprocess(self, requests):
        queries = [json.loads(req.get("data")) for req in requests]
        return queries
    
    def inference(self, queries):
        return self.model.infer_step(queries, top_k=5)
    
    def postprocess(self, results):
        return [json.dumps(results, default=str)]