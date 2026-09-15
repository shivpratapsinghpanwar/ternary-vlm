from .ternary import TernaryLoRALinear, ternary_quant, ternary_quant_ste, int8_act_quant_ste, wrap_linears, merge_all
from .ops import pixel_shuffle


def __getattr__(name):  # lazy: keep `import ternavlm` free of transformers for tests/tools
    if name in ("TernaVLM", "TernaVLMConfig"):
        from . import vlm
        return getattr(vlm, name)
    raise AttributeError(name)
