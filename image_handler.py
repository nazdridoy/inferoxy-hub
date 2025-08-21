"""
Image generation functionality handler for HF-Inferoxy AI Hub.
Handles text-to-image generation with multiple providers.
"""

import os
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from hf_token_utils import get_proxy_token, report_token_status
from utils import (
    IMAGE_CONFIG, 
    validate_proxy_key, 
    format_error_message, 
    format_success_message
)


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
):
    """
    Generate an image using the specified model and provider through HF-Inferoxy.
    """
    # Validate proxy API key
    is_valid, error_msg = validate_proxy_key()
    if not is_valid:
        return None, error_msg
    
    proxy_api_key = os.getenv("PROXY_KEY")
    
    try:
        # Get token from HF-Inferoxy proxy server
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
        print(f"📡 Image: Making generation request...")
        
        # Generate image
        image = client.text_to_image(**generation_params)
        
        print(f"🖼️ Image: Generation completed! Image type: {type(image)}")
        
        # Report successful token usage
        report_token_status(token_id, "success", api_key=proxy_api_key)
        
        return image, format_success_message("Image generated", f"using {model_name} on {provider}")
        
    except HfHubHTTPError as e:
        # Report HF Hub errors
        if 'token_id' in locals():
            report_token_status(token_id, "error", str(e), api_key=proxy_api_key)
        return None, format_error_message("HuggingFace API Error", str(e))
        
    except Exception as e:
        # Report other errors
        if 'token_id' in locals():
            report_token_status(token_id, "error", str(e), api_key=proxy_api_key)
        return None, format_error_message("Unexpected Error", str(e))


def handle_image_generation(prompt_val, model_val, provider_val, negative_prompt_val, width_val, height_val, steps_val, guidance_val, seed_val):
    """
    Handle image generation request with validation.
    """
    # Validate dimensions
    is_valid, error_msg = validate_dimensions(width_val, height_val)
    if not is_valid:
        return None, format_error_message("Validation Error", error_msg)
    
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
        seed=seed_val
    )
