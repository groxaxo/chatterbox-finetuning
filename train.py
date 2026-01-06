import os
import sys
import torch
from transformers import Trainer, TrainingArguments
from safetensors.torch import save_file

from src.config import TrainConfig
from src.dataset import ChatterboxDataset, data_collator
from src.model import resize_and_load_t3_weights, ChatterboxTrainerWrapper
from src.preprocess_ljspeech import preprocess_dataset_ljspeech
from src.preprocess_file_based import preprocess_dataset_file_based
from src.preprocess_json import preprocess_dataset_json_based
from src.utils import setup_logger, check_pretrained_models

from src.inference_callback import InferenceCallback

from src.chatterbox_.tts import ChatterboxTTS
from src.chatterbox_.tts_turbo import ChatterboxTurboTTS
from src.chatterbox_.models.t3.t3 import T3

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = setup_logger("ChatterboxFinetune")


def main():
    
    cfg = TrainConfig()
    
    logger.info("--- Starting Chatterbox Finetuning ---")
    logger.info(f"Mode: {'CHATTERBOX-TURBO' if cfg.is_turbo else 'CHATTERBOX-TTS'}")

    # 0. CHECK MODEL FILES
    mode_check = "chatterbox_turbo" if cfg.is_turbo else "chatterbox"
    if not check_pretrained_models(mode=mode_check):
        sys.exit(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. SELECT THE CORRECT ENGINE CLASS
    if cfg.is_turbo:
        EngineClass = ChatterboxTurboTTS
    else:
        EngineClass = ChatterboxTTS
    
    logger.info(f"Device: {device}")
    logger.info(f"Model Directory: {cfg.model_dir}")

    # 2. LOAD ORIGINAL MODEL TEMPORARILY
    logger.info("Loading original model to extract weights...")
    # Loading on CPU first to save VRAM
    tts_engine_original = EngineClass.from_local(cfg.model_dir, device="cpu")

    pretrained_t3_state_dict = tts_engine_original.t3.state_dict()
    original_t3_config = tts_engine_original.t3.hp

    # 3. CREATE NEW T3 MODEL WITH NEW VOCAB SIZE
    logger.info(f"Creating new T3 model with vocab size: {cfg.new_vocab_size}")
    
    new_t3_config = original_t3_config
    new_t3_config.text_tokens_dict_size = cfg.new_vocab_size

    # We prevent caching during training.
    if hasattr(new_t3_config, "use_cache"):
        new_t3_config.use_cache = False
    else:
        setattr(new_t3_config, "use_cache", False)

    new_t3_model = T3(hp=new_t3_config)

    # 4. TRANSFER WEIGHTS
    logger.info("Transferring weights...")
    new_t3_model = resize_and_load_t3_weights(new_t3_model, pretrained_t3_state_dict)


    # --- SPECIAL SETTING FOR TURBO ---
    if cfg.is_turbo:
        logger.info("Turbo Mode: Removing backbone WTE layer...")
        if hasattr(new_t3_model.tfmr, "wte"):
            del new_t3_model.tfmr.wte


    # Clean up memory
    del tts_engine_original
    del pretrained_t3_state_dict

    # 5. PREPARE ENGINE FOR TRAINING
    # Reload engine components (VoiceEncoder, S3Gen) but inject our new T3
    tts_engine_new = EngineClass.from_local(cfg.model_dir, device="cpu")
    tts_engine_new.t3 = new_t3_model 

    # Freeze other components
    logger.info("Freezing S3Gen and VoiceEncoder...")
    for param in tts_engine_new.ve.parameters(): 
        param.requires_grad = False
        
    for param in tts_engine_new.s3gen.parameters(): 
        param.requires_grad = False

    # Enable Training for T3
    tts_engine_new.t3.train()
    for param in tts_engine_new.t3.parameters(): 
        param.requires_grad = True

    if cfg.preprocess:
        
        logger.info("Initializing Preprocess dataset...")
        
        if cfg.ljspeech:
            preprocess_dataset_ljspeech(cfg, tts_engine_new)
            
        elif cfg.json_format:
            preprocess_dataset_json_based(cfg, tts_engine_new)
            
        else:
            preprocess_dataset_file_based(cfg, tts_engine_new)
      
    else:
        logger.info("Skipping the preprocessing dataset step...")
            
        
    # 6. DATASET & WRAPPER
    logger.info("Initializing Dataset...")
    train_ds = ChatterboxDataset(cfg)
    
    
    trainer_callbacks = []
    if cfg.is_inference:
        inference_cb = InferenceCallback(cfg)
        trainer_callbacks.append(inference_cb)
    
    model_wrapper = ChatterboxTrainerWrapper(tts_engine_new.t3)

    # 7. TRAINING ARGUMENTS
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_epochs,
        save_strategy="steps",
        save_steps=500,
        logging_strategy="epoch",
        remove_unused_columns=False, # Required for our custom wrapper
        dataloader_num_workers=4,    
        report_to=["tensorboard"],
        fp16=False,
        bf16=True,
        save_total_limit=2,
        gradient_checkpointing=True, # This setting theoretically reduces VRAM usage by 60%.
    )

    trainer = Trainer(
        model=model_wrapper,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
        callbacks=trainer_callbacks
    )

    logger.info("Starting Training Loop...")
    trainer.train()


    # 8. SAVE FINAL MODEL
    logger.info("Training complete. Saving model...")
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    filename = "t3_turbo_finetuned.safetensors" if cfg.is_turbo else "t3_finetuned.safetensors"
    final_model_path = os.path.join(cfg.output_dir, filename)

    save_file(tts_engine_new.t3.state_dict(), final_model_path)
    logger.info(f"Model saved to: {final_model_path}")


if __name__ == "__main__": 
    main()
