"""
Text-to-speech functionality handler for AI-Inferoxy AI Hub.
Handles text-to-speech generation with multiple providers.
"""

import os
import gradio as gr
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from requests.exceptions import ConnectionError, Timeout, RequestException
from hf_token_utils import get_proxy_token, report_token_status
from utils import (
    IMAGE_CONFIG, 
    validate_proxy_key, 
    format_error_message, 
    format_success_message,
    TTS_MODEL_CONFIGS,
)

# Timeout configuration for TTS generation
TTS_GENERATION_TIMEOUT = 300  # 5 minutes max for TTS generation


def generate_text_to_speech(
    text: str,
    model_name: str,
    provider: str,
    voice: str = "af_bella",
    speed: float = 1.0,
    audio_url: str = "",
    exaggeration: float = 0.25,
    temperature: float = 0.7,
    cfg: float = 0.5,
    client_name: str | None = None,
):
    """
    Generate speech from text using the specified model and provider through AI-Inferoxy.
    """
    # Validate proxy API key
    is_valid, error_msg = validate_proxy_key()
    if not is_valid:
        return None, error_msg
    
    proxy_api_key = os.getenv("PROXY_KEY")
    
    token_id = None
    try:
        # Get token from AI-Inferoxy proxy server with timeout handling
        print(f"🔑 TTS: Requesting token from proxy...")
        token, token_id = get_proxy_token(api_key=proxy_api_key)
        print(f"✅ TTS: Got token: {token_id}")
        
        print(f"🎤 TTS: Using model='{model_name}', provider='{provider}', voice='{voice}'")
        
        # Create client with specified provider
        client = InferenceClient(
            provider=provider,
            api_key=token
        )
        
        print(f"🚀 TTS: Client created, preparing generation params...")
        
        # Get model configuration
        model_config = TTS_MODEL_CONFIGS.get(model_name, {})
        extra_body_params = model_config.get("extra_body_params", [])
        
        # Prepare generation parameters
        generation_params = {
            "text": text,
            "model": model_name,
            "extra_body": {}
        }
        
        # Add model-specific parameters to extra_body
        if "voice" in extra_body_params:
            generation_params["extra_body"]["voice"] = voice
        if "speed" in extra_body_params:
            generation_params["extra_body"]["speed"] = speed
        if "audio_url" in extra_body_params:
            generation_params["extra_body"]["audio_url"] = audio_url
        if "exaggeration" in extra_body_params:
            generation_params["extra_body"]["exaggeration"] = exaggeration
        if "temperature" in extra_body_params:
            generation_params["extra_body"]["temperature"] = temperature
        if "cfg" in extra_body_params:
            generation_params["extra_body"]["cfg"] = cfg
        
        print(f"📡 TTS: Making generation request with {TTS_GENERATION_TIMEOUT}s timeout...")
        
        # Create generation function for timeout handling
        def generate_audio_task():
            return client.text_to_speech(**generation_params)
        
        # Execute with timeout using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(generate_audio_task)
            
            try:
                # Generate audio with timeout
                audio = future.result(timeout=TTS_GENERATION_TIMEOUT)
            except FutureTimeoutError:
                future.cancel()  # Cancel the running task
                raise TimeoutError(f"TTS generation timed out after {TTS_GENERATION_TIMEOUT} seconds")
        
        print(f"🎵 TTS: Generation completed! Audio type: {type(audio)}")
        
        # Report successful token usage
        if token_id:
            report_token_status(token_id, "success", api_key=proxy_api_key, client_name=client_name)
        
        return audio, format_success_message("Speech generated", f"using {model_name} on {provider} with voice {voice}")
        
    except ConnectionError as e:
        # Handle proxy connection errors
        error_msg = f"Cannot connect to AI-Inferoxy server: {str(e)}"
        print(f"🔌 TTS connection error: {error_msg}")
        if token_id:
            report_token_status(token_id, "error", error_msg, api_key=proxy_api_key, client_name=client_name)
        return None, format_error_message("Connection Error", "Unable to connect to the proxy server. Please check if it's running.")
        
    except TimeoutError as e:
        # Handle timeout errors
        error_msg = f"TTS generation timed out: {str(e)}"
        print(f"⏰ TTS timeout: {error_msg}")
        if token_id:
            report_token_status(token_id, "error", error_msg, api_key=proxy_api_key, client_name=client_name)
        return None, format_error_message("Timeout Error", f"TTS generation took too long (>{TTS_GENERATION_TIMEOUT//60} minutes). Try shorter text.")
        
    except HfHubHTTPError as e:
        # Handle HuggingFace API errors
        error_msg = str(e)
        print(f"🤗 TTS HF error: {error_msg}")
        if token_id:
            report_token_status(token_id, "error", error_msg, api_key=proxy_api_key, client_name=client_name)
        
        # Provide more user-friendly error messages
        if "401" in error_msg:
            return None, format_error_message("Authentication Error", "Invalid or expired API token. The proxy will provide a new token on retry.")
        elif "402" in error_msg:
            return None, format_error_message("Quota Exceeded", "API quota exceeded. The proxy will try alternative providers.")
        elif "429" in error_msg:
            return None, format_error_message("Rate Limited", "Too many requests. Please wait a moment and try again.")
        else:
            return None, format_error_message("HuggingFace API Error", error_msg)
        
    except Exception as e:
        # Handle all other errors
        error_msg = str(e)
        print(f"❌ TTS unexpected error: {error_msg}")
        if token_id:
            report_token_status(token_id, "error", error_msg, api_key=proxy_api_key)
        return None, format_error_message("Unexpected Error", f"An unexpected error occurred: {error_msg}")


def handle_text_to_speech_generation(text_val, model_val, provider_val, voice_val, speed_val, audio_url_val, exaggeration_val, temperature_val, cfg_val, hf_token: gr.OAuthToken = None):
    """
    Handle text-to-speech generation request with validation.
    """
    # Validate input text
    if not text_val or not text_val.strip():
        return None, format_error_message("Validation Error", "Please enter some text to convert to speech")
    
    # Limit text length to prevent timeouts
    if len(text_val) > 5000:
        return None, format_error_message("Validation Error", "Text is too long. Please keep it under 5000 characters.")
    
    # Require sign-in via HF OAuth token
    access_token = getattr(hf_token, "token", None) if hf_token is not None else None
    username = None
    if not access_token:
        return None, format_error_message("Access Required", "Please sign in with Hugging Face (sidebar Login button).")
    
    # Generate speech
    return generate_text_to_speech(
        text=text_val.strip(),
        model_name=model_val,
        provider=provider_val,
        voice=voice_val,
        speed=speed_val,
        audio_url=audio_url_val,
        exaggeration=exaggeration_val,
        temperature=temperature_val,
        cfg=cfg_val,
        client_name=username
    )
