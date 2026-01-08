import os
import json
import torch
import torchaudio
from tqdm import tqdm

from src.chatterbox_.tts import ChatterboxTTS, punc_norm
from src.chatterbox_.models.s3tokenizer import S3_SR
from src.utils import setup_logger


logger = setup_logger(__name__)


def preprocess_dataset_json_based(config, tts_engine: ChatterboxTTS):
    
    """
    Reads metadata from JSON file, processes audio-text pairs, and saves them as .pt.
    Structure:
    - JSON contains: id, text, formatted_text, etc.
    - Audio files: {wav_dir}/{id}.wav
    """
    
    os.makedirs(config.preprocessed_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tts_engine.ve.to(device)
    tts_engine.s3gen.to(device)
    tts_engine.ve.eval()
    tts_engine.s3gen.eval()
    

    if not os.path.exists(config.metadata_path):
        logger.error(f"ERROR: Metadata file not found: '{config.metadata_path}'!")
        return
    
    with open(config.metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    if len(metadata) == 0:
        logger.error(f"ERROR: No items found in metadata file!")
        return
    
    logger.info(f"Processing dataset... Found items in JSON: {len(metadata)}")
    
    success_count = 0
    
    for item in tqdm(metadata, desc="Preprocessing"):
        try:

            file_id = item.get("id")
            raw_text = item.get("text", "")
            
            if not file_id or not raw_text:
                logger.warning(f"Skipping item with missing id or text")
                continue
            

            wav_path = os.path.join(config.wav_dir, f"{file_id}.wav")
            
            if not os.path.exists(wav_path):
                logger.warning(f"Audio file not found, skipping: {file_id}")
                continue
            
            wav, sr = torchaudio.load(wav_path)
            
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            
            if sr != S3_SR:
                resampler = torchaudio.transforms.Resample(sr, S3_SR)
                wav = resampler(wav)
            
            wav = wav.to(device)
            
            with torch.no_grad():

                wav_np = wav.cpu().squeeze().numpy()
                spk_emb_np = tts_engine.ve.embeds_from_wavs([wav_np], sample_rate=S3_SR)
                speaker_emb = torch.from_numpy(spk_emb_np[0]).cpu()
                
                s_tokens, _ = tts_engine.s3gen.tokenizer(wav.unsqueeze(0))
                speech_tokens = s_tokens.squeeze().cpu()
                
                prompt_samples = int(config.prompt_duration * S3_SR)
                
                if wav.shape[1] < prompt_samples:
                    prompt_wav = torch.nn.functional.pad(wav, (0, prompt_samples - wav.shape[1]))
                else:
                    prompt_wav = wav[:, :prompt_samples]
                
                p_tokens, _ = tts_engine.s3gen.tokenizer(prompt_wav.unsqueeze(0))
                prompt_tokens = p_tokens.squeeze().cpu()
            
            clean_text = punc_norm(raw_text)
            
            if config.is_turbo:
                # For turbo mode, tokenize without special tokens and let dataset.py add them
                token_output = tts_engine.tokenizer(clean_text, return_tensors="pt", add_special_tokens=False)
                text_tokens = token_output.input_ids[0].cpu()
            
            else:
                text_tokens = tts_engine.tokenizer.text_to_tokens(clean_text).squeeze(0).cpu()
            
            save_path = os.path.join(config.preprocessed_dir, f"{file_id}.pt")
            
            torch.save({
                "speech_tokens": speech_tokens,
                "speaker_emb": speaker_emb,
                "prompt_tokens": prompt_tokens,
                "text_tokens": text_tokens,
            }, save_path)
            
            success_count += 1

            
        except Exception as e:
            logger.error(f"Error ({item.get('id', 'unknown')}): {e}")
            continue
    
    logger.info(f"Preprocessing completed! Success: {success_count}/{len(metadata)}")