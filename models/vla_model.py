import torch
import sys
import types
import transformers

# Transformers v5 compatibility: Create a synthetic tokenization_utils module
# that re-exports symbols from tokenization_utils_base (which moved/renamed in v5).
# Each symbol is loaded individually so one missing entry doesn't break the rest.
print("--- [Compatibility Fix] Injecting transformers.tokenization_utils ---")
_m = types.ModuleType("tokenization_utils")
_symbols_loaded = 0
_symbols_failed = 0
def _try_inject(src, name, fallback=None):
    global _symbols_loaded, _symbols_failed
    try:
        val = getattr(src, name)
        setattr(_m, name, val)
        _symbols_loaded += 1
    except AttributeError:
        if fallback is not None:
            setattr(_m, name, fallback)
            _symbols_loaded += 1
        else:
            _symbols_failed += 1

# Import symbols from tokenization_utils_base (available in both v4 and v5)
try:
    from transformers import tokenization_utils_base as _base
    for _sym in ["PaddingStrategy", "PreTokenizedInput", "TextInput",
                 "EncodedInput", "AddedToken", "TruncationStrategy",
                 "TextInputPair", "PreTokenizedInputPair", "EncodedInputPair"]:
        _try_inject(_base, _sym)
    # TextInputSequence: defined in v4's tokenization_utils_base, removed in v5.
    # It's a Union[str, List[str]] used in type hints — define manually.
    _try_inject(_base, "TextInputSequence", fallback=str)
except Exception as e:
    print(f"  [Warning] Could not access tokenization_utils_base: {e}")

sys.modules["transformers.tokenization_utils"] = _m
setattr(transformers, "tokenization_utils", _m)
print(f"--- [Compatibility Fix] Done — {_symbols_loaded} injected, {_symbols_failed} missing (non-critical) ---")

from transformers import AutoProcessor, AutoModel, AutoConfig, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Optional

class VLAModelWrapper:
    """
    Wrapper for loading and configuring VLA models with PEFT (QLoRA).
    Defaults to OpenVLA-like architecture.
    """
    
    def __init__(
        self, 
        model_id: str = "openvla/openvla-7b", 
        load_in_4bit: bool = True,
        device: str = "cuda"
    ):
        self.model_id = model_id
        self.device = device
        
        # 1. Setup 4-bit Quantization Config
        self.bnb_config = None
        if load_in_4bit:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        print(f"Loading model: {model_id}...")
        
        # Transformers v5 compatibility fix for legacy remote models
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

        if hasattr(config, "auto_map") and "AutoModel" not in config.auto_map:
            if "AutoModelForVision2Seq" in config.auto_map:
                config.auto_map["AutoModel"] = config.auto_map["AutoModelForVision2Seq"]

        # Monkey-patch tie_weights for Transformers v5 (Prismatic/OpenVLA fix)
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        try:
            model_class = get_class_from_dynamic_module(config.auto_map["AutoModel"], model_id)
            original_tie_weights = model_class.tie_weights
            def patched_tie_weights(self, *args, **kwargs):
                # Transformers v5 passes various kwargs (recompute_mapping, missing_keys)
                # Legacy code only expects self (and maybe positional args)
                return original_tie_weights(self, *args)
            model_class.tie_weights = patched_tie_weights
        except Exception as e:
            print(f"Warning: Could not patch tie_weights: {e}")

        self.model = AutoModel.from_pretrained(
            model_id,
            config=config,
            quantization_config=self.bnb_config,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            attn_implementation="eager"
        )
        
        # Note: Processor handling varies by model
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    def apply_lora(
        self, 
        r: int = 32, 
        alpha: int = 64, 
        target_modules: Optional[list] = None
    ):
        """
        Inject LoRA adapters into the model.
        """
        print("Applying LoRA adapters...")
        
        # Default target modules for Llama-based models (like OpenVLA's LLM)
        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM" # VLA is typically trained as causal LM on actions
        )

        # Prepare for 4-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        return self.model

    def save_checkpoint(self, path: str):
        """
        Saves only the LoRA adapters and config.
        """
        self.model.save_pretrained(path)
        print(f"Checkpoint saved to {path}")

if __name__ == "__main__":
    # Mock initialization (will fail without model access/gpu but shows logic)
    try:
        wrapper = VLAModelWrapper(model_id="openvla/openvla-7b", load_in_4bit=True)
        model = wrapper.apply_lora()
    except Exception as e:
        print(f"Initialization skipped/failed: {e}")
