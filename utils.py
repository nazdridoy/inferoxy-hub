"""
Utility functions and constants for HF-Inferoxy AI Hub.
Contains configuration constants and helper functions.
"""

import os


# Configuration constants
DEFAULT_CHAT_MODEL = "openai/gpt-oss-20b"
DEFAULT_IMAGE_MODEL = "Qwen/Qwen-Image"
DEFAULT_IMAGE_PROVIDER = "fal-ai"

# Chat configuration
CHAT_CONFIG = {
    "max_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.95,
    "system_message": "You are a helpful and friendly AI assistant. Provide clear, accurate, and helpful responses."
}

# Image generation configuration
IMAGE_CONFIG = {
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 20,
    "guidance_scale": 7.5,
    "seed": -1,
    "negative_prompt": "blurry, low quality, distorted, deformed, ugly, bad anatomy"
}

# Supported providers
CHAT_PROVIDERS = ["auto", "fireworks-ai", "cerebras", "groq", "together", "cohere"]
IMAGE_PROVIDERS = ["hf-inference", "fal-ai", "nebius", "nscale", "replicate", "together"]

# Popular models for quick access
POPULAR_CHAT_MODELS = [
    "openai/gpt-oss-20b",
    "meta-llama/Llama-2-7b-chat-hf", 
    "microsoft/DialoGPT-medium",
    "google/flan-t5-base"
]

POPULAR_IMAGE_MODELS = [
    "Qwen/Qwen-Image",
    "black-forest-labs/FLUX.1-dev",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "runwayml/stable-diffusion-v1-5"
]

# Model presets for image generation
IMAGE_MODEL_PRESETS = [
    ("Qwen (Fal.ai)", "Qwen/Qwen-Image", "fal-ai"),
    ("Qwen (Replicate)", "Qwen/Qwen-Image", "replicate"),
    ("FLUX.1 (Nebius)", "black-forest-labs/FLUX.1-dev", "nebius"), 
    ("SDXL (HF)", "stabilityai/stable-diffusion-xl-base-1.0", "hf-inference"),
]

# Example prompts for image generation
IMAGE_EXAMPLE_PROMPTS = [
    "A majestic dragon flying over a medieval castle, epic fantasy art, detailed, 8k",
    "A serene Japanese garden with cherry blossoms, zen atmosphere, peaceful, high quality",
    "A futuristic cityscape with flying cars and neon lights, cyberpunk style, cinematic",
    "A cute robot cat playing with yarn, adorable, cartoon style, vibrant colors",
    "A magical forest with glowing mushrooms and fairy lights, fantasy, ethereal beauty",
    "Portrait of a wise old wizard with flowing robes, magical aura, fantasy character art",
    "A cozy coffee shop on a rainy day, warm lighting, peaceful atmosphere, detailed",
    "An astronaut floating in space with Earth in background, photorealistic, stunning"
]


def get_proxy_key():
    """Get the proxy API key from environment variables."""
    return os.getenv("PROXY_KEY")


def validate_proxy_key():
    """Validate that the proxy key is available."""
    proxy_key = get_proxy_key()
    if not proxy_key:
        return False, "❌ Error: PROXY_KEY not found in environment variables. Please set it in your HuggingFace Space secrets."
    return True, ""


def get_proxy_url():
    """Get the proxy URL from environment variables."""
    return os.getenv("PROXY_URL")


def validate_proxy_url():
    """Validate that the proxy URL is available."""
    proxy_url = get_proxy_url()
    if not proxy_url:
        return False, "❌ Error: PROXY_URL not found in environment variables. Please set it in your HuggingFace Space secrets."
    return True, ""


def parse_model_and_provider(model_name):
    """
    Parse model name and provider from a string like 'model:provider'.
    Returns (model, provider) tuple. Provider is None if not specified.
    """
    if ":" in model_name:
        model, provider = model_name.split(":", 1)
        return model, provider
    else:
        return model_name, None


def format_error_message(error_type, error_message):
    """Format error messages consistently."""
    return f"❌ {error_type}: {error_message}"


def format_success_message(operation, details=""):
    """Format success messages consistently."""
    base_message = f"✅ {operation} completed successfully"
    if details:
        return f"{base_message}: {details}"
    return f"{base_message}!"


def get_gradio_theme():
    """Get the default Gradio theme for the application."""
    try:
        import gradio as gr
        return gr.themes.Soft()
    except ImportError:
        return None
