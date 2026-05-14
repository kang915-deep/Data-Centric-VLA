import torch
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
        
        # Module hijacking for PaddingStrategy in Transformers v5
        import sys
        import types
        try:
            from transformers.utils import PaddingStrategy
            if "transformers.tokenization_utils" not in sys.modules:
                mock_module = types.ModuleType("transformers.tokenization_utils")
                mock_module.PaddingStrategy = PaddingStrategy
                sys.modules["transformers.tokenization_utils"] = mock_module
        except ImportError:
            pass

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
