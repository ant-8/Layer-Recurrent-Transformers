from transformers import GPT2Config

class LoopedGPT2Config(GPT2Config):

    def __init__(
        self,
        num_loops=1,
        positional_encoding="ape",
        sequential_looping=False,
        loop_all_layers=False,
        loop_map=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_type = "LoopedGPT2ForCausalLM"
        self.num_loops = num_loops
        self.sequential_looping = sequential_looping
        self.loop_all_layers = loop_all_layers
        self.loop_map = loop_map
        self.positional_encoding=positional_encoding

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        config_dict = config_dict.copy()
        num_loops = config_dict.pop("num_loops", 1)
        sequential_looping = config_dict.pop("sequential_looping", False)
        loop_all_layers = config_dict.pop("loop_all_layers", False)
        loop_map = config_dict.pop("loop_map", None)
        positional_encoding = config_dict.pop("positional_encoding", None)

        base_config = super().from_dict(config_dict, **kwargs)

        base_config.num_loops = num_loops
        base_config.sequential_looping = sequential_looping
        base_config.loop_all_layers = loop_all_layers
        base_config.loop_map = loop_map
        base_config.positional_encoding = positional_encoding

        return base_config

    def to_dict(self):
        config_dict = super().to_dict()
        config_dict["num_loops"] = self.num_loops
        config_dict["sequential_looping"] = self.sequential_looping
        config_dict["loop_all_layers"] = self.loop_all_layers
        config_dict["loop_map"] = self.loop_map
        config_dict["positional_encoding"] = self.positional_encoding
        return config_dict