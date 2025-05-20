from transformers import LlamaConfig

class LoopedLlamaConfig(LlamaConfig):

    def __init__(
        self,
        loop_map = None,
        sequential_looping=False,
        num_loops = None,
        intermediate_size_map = None,
        intermediate_layer_map = None,
        use_head_scale=False,
        num_intermediate_layers = 1,
        positional_encoding = "rope",
        use_recurrent_embeds = False,
        use_loop_encoding=False,
        use_adaptive_layer_norm=False,
        attn_scaling=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_type = "LoopedLlamaForCausalLM"
        self.loop_map = loop_map
        self.sequential_looping = sequential_looping
        self.num_loops = num_loops
        self.intermediate_size_map = intermediate_size_map
        self.intermediate_layer_map = intermediate_layer_map
        self.num_intermediate_layers = num_intermediate_layers
        self.attn_scaling = attn_scaling
        self.use_head_scale = use_head_scale
        self.positional_encoding = positional_encoding
        self.use_loop_encoding = use_loop_encoding
        self.use_adaptive_layer_norm = use_adaptive_layer_norm
        self.use_recurrent_embeds = use_recurrent_embeds
    
    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        config_dict = config_dict.copy()
        loop_map = config_dict.pop("loop_map", None)
        sequential_looping = config_dict.pop("sequential_looping", False)
        num_loops = config_dict.pop("num_loops", False)
        intermediate_size_map = config_dict.pop("intermediate_size_map", None)
        intermediate_layer_map = config_dict.pop("intermediate_layer_map", None)
        num_intermediate_layers = config_dict.pop("num_intermediate_layers", 1)
        attn_scaling = config_dict.pop("attn_scaling", None)
        positional_encoding = config_dict.pop("positional_encoding", None)
        
        base_config = super().from_dict(config_dict, **kwargs)

        base_config.loop_map = loop_map
        base_config.num_loops = num_loops
        base_config.intermediate_size_map = intermediate_size_map
        base_config.intermediate_layer_map = intermediate_layer_map
        base_config.sequential_looping = sequential_looping
        base_config.num_intermediate_layers = num_intermediate_layers
        base_config.attn_scaling = attn_scaling
        base_config.positional_encoding = positional_encoding
        
        return base_config

    def to_dict(self):
        config_dict = super().to_dict()
        config_dict["loop_map"] = self.loop_map
        config_dict["sequential_looping"] = self.sequential_looping
        config_dict["num_loops"] = self.num_loops
        config_dict["intermediate_size_map"] = self.intermediate_size_map
        config_dict["num_intermediate_layers"] = self.num_intermediate_layers
        config_dict["intermediate_layer_map"] = self.intermediate_layer_map
        config_dict["attn_scaling"] = self.attn_scaling
        config_dict["positional_encoding"] = self.positional_encoding
        return config_dict
