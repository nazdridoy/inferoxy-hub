"""
Text-to-video functionality handler for AI-Inferoxy AI Hub.
Handles text-to-video generation with multiple providers.
"""

import os
import gradio as gr
import tempfile
import io
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from requests.exceptions import ConnectionError
from hf_token_utils import get_proxy_token, report_token_status
from utils import (
    validate_proxy_key,
    format_error_message,
    format_success_message,
)


# Timeout configuration for video generation
VIDEO_GENERATION_TIMEOUT = 600  # up to 10 minutes, videos can be slow


def generate_video(
    prompt: str,
    model_name: str,
    provider: str,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    seed: int | None = None,
    client_name: str | None = None,
):
    """
    Generate a video using the specified model and provider through AI-Inferoxy.
    Returns (video_bytes_or_url, status_message)
    """
    # Validate proxy API key
    is_valid, error_msg = validate_proxy_key()
    if not is_valid:
        return None, error_msg

    proxy_api_key = os.getenv("PROXY_KEY")

    token_id = None
    try:
        # Get token from AI-Inferoxy proxy server with timeout handling
        print(f"🔑 Video: Requesting token from proxy...")
        token, token_id = get_proxy_token(api_key=proxy_api_key)
        print(f"✅ Video: Got token: {token_id}")

        print(f"🎬 Video: Using model='{model_name}', provider='{provider}'")

        # Create client with specified provider
        client = InferenceClient(
            provider=provider,
            api_key=token
        )

        # Prepare generation parameters
        generation_params: dict = {
            "model": model_name,
            "prompt": prompt,
        }
        if num_inference_steps is not None:
            generation_params["num_inference_steps"] = num_inference_steps
        if guidance_scale is not None:
            generation_params["guidance_scale"] = guidance_scale
        if seed is not None and seed != -1:
            generation_params["seed"] = seed

        print(f"📡 Video: Making generation request with {VIDEO_GENERATION_TIMEOUT}s timeout...")

        # Create generation function for timeout handling
        def generate_video_task():
            return client.text_to_video(**generation_params)

        # Execute with timeout using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(generate_video_task)
            try:
                video = future.result(timeout=VIDEO_GENERATION_TIMEOUT)
            except FutureTimeoutError:
                future.cancel()
                raise TimeoutError(f"Video generation timed out after {VIDEO_GENERATION_TIMEOUT} seconds")

        print(f"🎞️ Video: Generation completed! Type: {type(video)}")

        # Convert output to a path or URL Gradio can handle
        video_output = _coerce_video_output(video)

        # Report successful token usage
        if token_id:
            report_token_status(token_id, "success", api_key=proxy_api_key, client_name=client_name)

        return video_output, format_success_message("Video generated", f"using {model_name} on {provider}")

    except ConnectionError as e:
        error_msg = f"Cannot connect to AI-Inferoxy server: {str(e)}"
        print(f"🔌 Video connection error: {error_msg}")
        if token_id:
            report_token_status(token_id, "error", error_msg, api_key=proxy_api_key, client_name=client_name)
        return None, format_error_message("Connection Error", "Unable to connect to the proxy server. Please check if it's running.")

    except TimeoutError as e:
        error_msg = f"Video generation timed out: {str(e)}"
        print(f"⏰ Video timeout: {error_msg}")
        if token_id:
            report_token_status(token_id, "error", error_msg, api_key=proxy_api_key, client_name=client_name)
        return None, format_error_message("Timeout Error", f"Video generation took too long (>{VIDEO_GENERATION_TIMEOUT//60} minutes). Try a shorter prompt.")

    except HfHubHTTPError as e:
        error_msg = str(e)
        print(f"🤗 Video HF error: {error_msg}")
        if token_id:
            report_token_status(token_id, "error", error_msg, api_key=proxy_api_key, client_name=client_name)
        if "401" in error_msg:
            return None, format_error_message("Authentication Error", "Invalid or expired API token. The proxy will provide a new token on retry.")
        elif "402" in error_msg:
            return None, format_error_message("Quota Exceeded", "API quota exceeded. The proxy will try alternative providers.")
        elif "429" in error_msg:
            return None, format_error_message("Rate Limited", "Too many requests. Please wait a moment and try again.")
        else:
            return None, format_error_message("HuggingFace API Error", error_msg)

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Video unexpected error: {error_msg}")
        if token_id:
            report_token_status(token_id, "error", error_msg, api_key=proxy_api_key)
        return None, format_error_message("Unexpected Error", f"An unexpected error occurred: {error_msg}")


def handle_video_generation(prompt_val, model_val, provider_val, steps_val, guidance_val, seed_val, hf_token: gr.OAuthToken = None):
    """
    Handle text-to-video generation request with validation and org access.
    """
    if not prompt_val or not prompt_val.strip():
        return None, format_error_message("Validation Error", "Please enter a prompt for video generation")

    access_token = getattr(hf_token, "token", None) if hf_token is not None else None
    username = None
    if not access_token:
        return None, format_error_message("Access Required", "Please sign in with Hugging Face (sidebar Login button).")

    return generate_video(
        prompt=prompt_val.strip(),
        model_name=model_val,
        provider=provider_val,
        num_inference_steps=steps_val if steps_val is not None else None,
        guidance_scale=guidance_val if guidance_val is not None else None,
        seed=seed_val if seed_val is not None else None,
        client_name=username,
    )


def _coerce_video_output(value):
    """Coerce various return types (bytes, str path/URL, BytesIO) into a filepath/URL for gr.Video."""
    # Case 1: Direct URL or existing file path
    if isinstance(value, str):
        if value.startswith("http://") or value.startswith("https://"):
            return value
        if os.path.exists(value):
            return value
        # Unknown string; fall through to save as file

    # Case 2: Bytes-like content
    if isinstance(value, (bytes, bytearray)):
        data = bytes(value)
        suffix = _guess_video_suffix(data)
        return _write_temp_video(data, suffix)

    # Case 3: File-like object
    if isinstance(value, io.IOBase) or hasattr(value, "read"):
        try:
            data = value.read()
            if isinstance(data, (bytes, bytearray)):
                suffix = _guess_video_suffix(data)
                return _write_temp_video(bytes(data), suffix)
        except Exception:
            pass

    # Fallback: save string representation for debugging
    debug_bytes = str(type(value)).encode("utf-8")
    return _write_temp_video(debug_bytes, ".mp4")


def _write_temp_video(data: bytes, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(data)
        tmp.flush()
    finally:
        tmp.close()
    return tmp.name


def _guess_video_suffix(data: bytes) -> str:
    header = data[:64]
    # MP4 often contains 'ftyp' box near start
    if b"ftyp" in header:
        return ".mp4"
    # WebM/Matroska magic number starts with 0x1A45DFA3 and often contains 'webm'
    if header.startswith(b"\x1aE\xdf\xa3") or b"webm" in header.lower():
        return ".webm"
    # Default to mp4
    return ".mp4"


