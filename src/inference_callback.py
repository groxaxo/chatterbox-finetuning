import os
import torch
import soundfile as sf
from transformers import TrainerCallback
from safetensors.torch import load_file

from src.chatterbox_.tts import ChatterboxTTS
from src.chatterbox_.tts_turbo import ChatterboxTurboTTS
from src.chatterbox_.models.t3.t3 import T3
from src.utils import setup_logger, trim_silence_with_vad


logger = setup_logger("InferenceCallback")


class InferenceCallback(TrainerCallback):
    
    def __init__(self, config):
        
        self.config = config
        self.inference_dir = os.path.join(config.output_dir, "inference_samples")
        os.makedirs(self.inference_dir, exist_ok=True)
        

        if not hasattr(config, 'inference_prompt_path') or not config.inference_prompt_path:
            logger.warning("The inference prompt path is not specified; sampling will be skipped.")
            self.skip_inference = True
            
        elif not hasattr(config, 'inference_test_text') or not config.inference_test_text:
            logger.warning("The inference test text is not specified; the sample will be skipped.")
            self.skip_inference = True
            
        else:
            self.skip_inference = False
            logger.info(f"Inference Callback is ready. Examples will be saved here: {self.inference_dir}")


    def on_save(self, args, state, control, **kwargs):

        if self.skip_inference:
            return
        
        step = state.global_step
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
        
        weights_path = os.path.join(checkpoint_dir, "model.safetensors")
        if not os.path.exists(weights_path):
            weights_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        
        if not os.path.exists(weights_path):
            logger.warning(f"Checkpoint weights could not be found: {checkpoint_dir}")
            return

        logger.info(f"Initializing inference for checkpoint-{step}...")


        try:
            
            output_path = os.path.join(self.inference_dir, f"checkpoint-{step}.wav")
            self._generate_sample(weights_path, output_path)
            
        except Exception as e:
            logger.error(f"An error occurred during the inference (Step: {step}): {e}", exc_info=True)



    def _generate_sample(self, checkpoint_path: str, output_path: str):

        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_turbo = getattr(self.config, "is_turbo", False)
        
        EngineClass = ChatterboxTurboTTS if is_turbo else ChatterboxTTS
        
        tts_engine = EngineClass.from_local(self.config.model_dir, device="cpu")
        
        t3_config = tts_engine.t3.hp
        if hasattr(self.config, 'new_vocab_size'):
            t3_config.text_tokens_dict_size = self.config.new_vocab_size
        
        new_t3 = T3(hp=t3_config)
        

        if is_turbo:
            if hasattr(new_t3.tfmr, "wte"):
                del new_t3.tfmr.wte


        if checkpoint_path.endswith(".safetensors"):
            state_dict = load_file(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")


        clean_state_dict = {}
        for k, v in state_dict.items():
            k_clean = k.replace("module.", "")
            
            if k_clean.startswith("t3."):
                clean_state_dict[k_clean.replace("t3.", "")] = v
            
            elif not any(x in k_clean for x in ["s3gen", "ve.", "tokenizer"]):
                clean_state_dict[k_clean] = v


        missing_keys, unexpected_keys = new_t3.load_state_dict(clean_state_dict, strict=False)
        
        
        critical_missing = [k for k in missing_keys if "tfmr.layers" in k]
        
        
        if len(critical_missing) > 0:
            logger.error("[CRITICAL ERROR] Model weights COULD NOT BE LOADED!")
            logger.error(f"Number of missing keys: {len(missing_keys)}")
            logger.error(f"Examples of missing information: {critical_missing[:3]}")
            logger.error("The sound produced will be 100% NOISE (Static Noise). Check your checkpoint recording method.")
        
        elif len(missing_keys) > 0:
            
            non_wte_missing = [k for k in missing_keys if "wte" not in k]
            if len(non_wte_missing) > 0:
                logger.warning(f"Some weights are missing ({len(non_wte_missing)} pieces): {non_wte_missing[:3]}...")
            
            else:
                logger.info("The weights were successfully loaded (except for the WTE - normal for the Turbo).")
        
        else:
            logger.info("All the weights were loaded completely and successfully.")


        tts_engine.t3 = new_t3
        tts_engine.to(device)
        tts_engine.eval()
        

        params = {
            "temperature": 0.8,
            "exaggeration": 0.5 if is_turbo else 1.2,
            "repetition_penalty": 1.2,
        }


        if not is_turbo:
            params["cfg_weight"] = 0.5


        with torch.no_grad():
            wav = tts_engine.generate(
                text=self.config.inference_test_text,
                audio_prompt_path=self.config.inference_prompt_path,
                **params
            )
        
        
        wav_np = wav.squeeze().cpu().numpy()
        trimmed_wav = trim_silence_with_vad(wav_np, tts_engine.sr)
        
        
        sf.write(output_path, trimmed_wav, tts_engine.sr)
        logger.info(f"Example saved: {output_path}")
        
        
        del tts_engine
        del new_t3
        del state_dict
        del clean_state_dict
        torch.cuda.empty_cache()