"""
Image generation functionality handler for AI-Inferoxy AI Hub.
Handles text-to-image generation with multiple providers.
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
)

# Timeout configuration for image generation
IMAGE_GENERATION_TIMEOUT = 300  # 5 minutes max for image generation


def validate_dimensions(width, height):
    """Validate that dimensions are divisible by 8 (required by most diffusion models)"""
    if width % 8 != 0 or height % 8 != 0:
        return False, "Width and height must be divisible by 8"
    return True, ""


def generate_image(
    prompt: str,
    model_name: str,
    provider: str,
    negative_prompt: str = "",
    width: int = IMAGE_CONFIG["width"],
    height: int = IMAGE_CONFIG["height"],
    num_inference_steps: int = IMAGE_CONFIG["num_inference_steps"],
    guidance_scale: float = IMAGE_CONFIG["guidance_scale"],
    seed: int = IMAGE_CONFIG["seed"],
    client_name: str | None = None,
):
    """
    Generate an image using the specified model and provider through AI-Inferoxy.
    """
    # Validate proxy API key
    is_valid, error_msg = validate_proxy_key()
    if not is_valid:
        return None, error_msg
    
    proxy_api_key = os.getenv("PROXY_KEY")
    
    token_id = None
    try:
        # Get token from AI-Inferoxy proxy server with timeout handling
        print(f"🔑 Image: Requesting token from proxy...")
        token, token_id = get_proxy_token(api_key=proxy_api_key)
        print(f"✅ Image: Got token: {token_id}")
        
        print(f"🎨 Image: Using model='{model_name}', provider='{provider}'")
        
        # Create client with specified provider
        client = InferenceClient(
            provider=provider,
            api_key=token
        )
        
        print(f"🚀 Image: Client created, preparing generation params...")
        
        # Prepare generation parameters
        generation_params = {
            "model": model_name,
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }
        
        # Add optional parameters if provided
        if negative_prompt:
            generation_params["negative_prompt"] = negative_prompt
        if seed != -1:
            generation_params["seed"] = seed
        
        print(f"📐 Image: Dimensions: {width}x{height}, steps: {num_inference_steps}, guidance: {guidance_scale}")
        print(f"📡 Image: Making generation request with {IMAGE_GENERATION_TIMEOUT}s timeout...")
        
        # Create generation function for timeout handling
        def generate_image_task():
            return client.text_to_image(**generation_params)
        
        # Execute with timeout using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(generate_image_task)
            
            try:
                # Generate image with timeout
                image = future.result(timeout=IMAGE_GENERATION_TIMEOUT)
            except FutureTimeoutError:
                future.cancel()  # Cancel the running task
                raise TimeoutError(f"Image generation timed out after {IMAGE_GENERATION_TIMEOUT} seconds")
        
        print(f"🖼️ Image: Generation completed! Image type: {type(image)}")
        
        # Report successful token usage
        if token_id:
            report_token_status(token_id, "success", api_key=proxy_api_key, client_name=client_name)
        
        return image, format_success_message("Image generated", f"using {model_name} on {provider}")
        
    except ConnectionError as e:
        # Handle proxy connection errors
        error_msg = f"Cannot connect to AI-Inferoxy server: {str(e)}"
        print(f"🔌 Image connection error: {error_msg}")
        if token_id:
            report_token_status(token_id, "error", error_msg, api_key=proxy_api_key, client_name=client_name)
        return None, format_error_message("Connection Error", "Unable to connect to the proxy server. Please check if it's running.")
        
    except TimeoutError as e:
        # Handle timeout errors
        error_msg = f"Image generation timed out: {str(e)}"
        print(f"⏰ Image timeout: {error_msg}")
        if token_id:
            report_token_status(token_id, "error", error_msg, api_key=proxy_api_key, client_name=client_name)
        return None, format_error_message("Timeout Error", f"Image generation took too long (>{IMAGE_GENERATION_TIMEOUT//60} minutes). Try reducing image size or steps.")
        
    except HfHubHTTPError as e:
        # Handle HuggingFace API errors
        error_msg = str(e)
        print(f"🤗 Image HF error: {error_msg}")
        if token_id:
            report_token_status(token_id, "error", error_msg, api_key=proxy_api_key, client_name=client_name)
        
        # Provide more user-friendly error messages
        if "401" in error_msg:
            return None, format_error_message("Authentication Error", "Invalid or expired API token. The proxy will provide a new token on retry.")
        elif "402" in error_msg:
            return None, format_error_message("Quota Exceeded", "API quota exceeded. The proxy will try alternative providers.")
        elif "429" in error_msg:
            return None, format_error_message("Rate Limited", "Too many requests. Please wait a moment and try again.")
        elif "content policy" in error_msg.lower() or "safety" in error_msg.lower():
            return None, format_error_message("Content Policy", "Image prompt was rejected by content policy. Please try a different prompt.")
        else:
            return None, format_error_message("HuggingFace API Error", error_msg)
        
    except Exception as e:
        # Handle all other errors
        error_msg = str(e)
        print(f"❌ Image unexpected error: {error_msg}")
        if token_id:
            report_token_status(token_id, "error", error_msg, api_key=proxy_api_key)
        return None, format_error_message("Unexpected Error", f"An unexpected error occurred: {error_msg}")


def generate_image_to_image(
    input_image,
    prompt: str,
    model_name: str,
    provider: str,
    negative_prompt: str = "",
    num_inference_steps: int = IMAGE_CONFIG["num_inference_steps"],
    guidance_scale: float = IMAGE_CONFIG["guidance_scale"],
    seed: int = IMAGE_CONFIG["seed"],
    client_name: str | None = None,
):
    """
    Generate an image using image-to-image generation with the specified model and provider through AI-Inferoxy.
    """
    # Validate proxy API key
    is_valid, error_msg = validate_proxy_key()
    if not is_valid:
        return None, error_msg
    
    proxy_api_key = os.getenv("PROXY_KEY")
    
    token_id = None
    try:
        # Get token from AI-Inferoxy proxy server with timeout handling
        print(f"🔑 Image-to-Image: Requesting token from proxy...")
        token, token_id = get_proxy_token(api_key=proxy_api_key)
        print(f"✅ Image-to-Image: Got token: {token_id}")
        
        print(f"🎨 Image-to-Image: Using model='{model_name}', provider='{provider}'")
        
        # Create client with specified provider
        client = InferenceClient(
            provider=provider,
            api_key=token
        )
        
        print(f"🚀 Image-to-Image: Client created, preparing generation params...")
        
        # Prepare generation parameters
        generation_params = {
            "image": input_image,
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }
        
        # Add optional parameters if provided
        if negative_prompt:
            generation_params["negative_prompt"] = negative_prompt
        if seed != -1:
            generation_params["seed"] = seed
        
        print(f"📡 Image-to-Image: Making generation request with {IMAGE_GENERATION_TIMEOUT}s timeout...")
        
        # Create generation function for timeout handling
        def generate_image_task():
            return client.image_to_image(
                model=model_name,
                **generation_params
            )
        
        # Execute with timeout using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(generate_image_task)
            
            try:
                # Generate image with timeout
                image = future.result(timeout=IMAGE_GENERATION_TIMEOUT)
            except FutureTimeoutError:
                future.cancel()  # Cancel the running task
                raise TimeoutError(f"Image-to-image generation timed out after {IMAGE_GENERATION_TIMEOUT} seconds")
        
        print(f"🖼️ Image-to-Image: Generation completed! Image type: {type(image)}")
        
        # Report successful token usage
        if token_id:
            report_token_status(token_id, "success", api_key=proxy_api_key, client_name=client_name)
        
        return image, format_success_message("Image-to-image generated", f"using {model_name} on {provider}")
        
    except ConnectionError as e:
        # Handle proxy connection errors
        error_msg = f"Cannot connect to AI-Inferoxy server: {str(e)}"
        print(f"🔌 Image-to-Image connection error: {error_msg}")
        if token_id:
            report_token_status(token_id, "error", error_msg, api_key=proxy_api_key, client_name=client_name)
        return None, format_error_message("Connection Error", "Unable to connect to the proxy server. Please check if it's running.")
        
    except TimeoutError as e:
        # Handle timeout errors
        error_msg = f"Image-to-image generation timed out: {str(e)}"
        print(f"⏰ Image-to-Image timeout: {error_msg}")
        if token_id:
            report_token_status(token_id, "error", error_msg, api_key=proxy_api_key, client_name=client_name)
        return None, format_error_message("Timeout Error", f"Image-to-image generation took too long (>{IMAGE_GENERATION_TIMEOUT//60} minutes). Try reducing steps.")
        
    except HfHubHTTPError as e:
        # Handle HuggingFace API errors
        error_msg = str(e)
        print(f"🤗 Image-to-Image HF error: {error_msg}")
        if token_id:
            report_token_status(token_id, "error", error_msg, api_key=proxy_api_key, client_name=client_name)
        
        # Provide more user-friendly error messages
        if "401" in error_msg:
            return None, format_error_message("Authentication Error", "Invalid or expired API token. The proxy will provide a new token on retry.")
        elif "402" in error_msg:
            return None, format_error_message("Quota Exceeded", "API quota exceeded. The proxy will try alternative providers.")
        elif "429" in error_msg:
            return None, format_error_message("Rate Limited", "Too many requests. Please wait a moment and try again.")
        elif "content policy" in error_msg.lower() or "safety" in error_msg.lower():
            return None, format_error_message("Content Policy", "Image prompt was rejected by content policy. Please try a different prompt.")
        else:
            return None, format_error_message("HuggingFace API Error", error_msg)
        
    except Exception as e:
        # Handle all other errors
        error_msg = str(e)
        print(f"❌ Image-to-Image unexpected error: {error_msg}")
        if token_id:
            report_token_status(token_id, "error", error_msg, api_key=proxy_api_key)
        return None, format_error_message("Unexpected Error", f"An unexpected error occurred: {error_msg}")


def handle_image_to_image_generation(input_image_val, prompt_val, model_val, provider_val, negative_prompt_val, steps_val, guidance_val, seed_val, hf_token: gr.OAuthToken = None):
    """
    Handle image-to-image generation request with validation.
    """
    # Validate input image
    if input_image_val is None:
        return None, format_error_message("Validation Error", "Please upload an input image")
    
    # Require sign-in via HF OAuth token
    access_token = getattr(hf_token, "token", None) if hf_token is not None else None
    username = None
    if not access_token:
        return None, format_error_message("Access Required", "Please sign in with Hugging Face (sidebar Login button).")
    
    # Generate image-to-image
    return generate_image_to_image(
        input_image=input_image_val,
        prompt=prompt_val,
        model_name=model_val,
        provider=provider_val,
        negative_prompt=negative_prompt_val,
        num_inference_steps=steps_val,
        guidance_scale=guidance_val,
        seed=seed_val,
        client_name=username
    )


def handle_image_generation(prompt_val, model_val, provider_val, negative_prompt_val, width_val, height_val, steps_val, guidance_val, seed_val, hf_token: gr.OAuthToken = None):
    """
    Handle image generation request with validation.
    """
    # Validate dimensions
    is_valid, error_msg = validate_dimensions(width_val, height_val)
    if not is_valid:
        return None, format_error_message("Validation Error", error_msg)
    
    # Require sign-in via HF OAuth token
    access_token = getattr(hf_token, "token", None) if hf_token is not None else None
    username = None
    if not access_token:
        return None, format_error_message("Access Required", "Please sign in with Hugging Face (sidebar Login button).")
    
    # Generate image
    return generate_image(
        prompt=prompt_val,
        model_name=model_val,
        provider=provider_val,
        negative_prompt=negative_prompt_val,
        width=width_val,
        height=height_val,
        num_inference_steps=steps_val,
        guidance_scale=guidance_val,
        seed=seed_val,
        client_name=username
    )
