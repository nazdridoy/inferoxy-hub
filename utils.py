"""
Utility functions and constants for HF-Inferoxy AI Hub.
Contains configuration constants and helper functions.
"""

import os
import re
import requests


# Configuration constants
DEFAULT_CHAT_MODEL = "openai/gpt-oss-20b"
DEFAULT_IMAGE_MODEL = "Qwen/Qwen-Image"
DEFAULT_IMAGE_TO_IMAGE_MODEL = "Qwen/Qwen-Image-Edit"
DEFAULT_TTS_MODEL = "hexgrad/Kokoro-82M"
DEFAULT_VIDEO_MODEL = "Wan-AI/Wan2.2-T2V-A14B"

# Unified default provider used by all non-chat tasks
DEFAULT_PROVIDER = "auto"

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

# Supported providers (unified across tasks)
PROVIDERS_UNIFIED = [
    "auto",
    "cerebras",
    "cohere",
    "fal-ai",
    "featherless-ai",
    "fireworks-ai",
    "groq",
    "hf-inference",
    "hyperbolic",
    "nebius",
    "novita",
    "nscale",
    "replicate",
    "sambanova",
    "together",
]

# Backwards compatibility exported lists
CHAT_PROVIDERS = PROVIDERS_UNIFIED
IMAGE_PROVIDERS = PROVIDERS_UNIFIED

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

# Suggested model lists (users can still input any model id)
SUGGESTED_CHAT_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "deepseek-ai/DeepSeek-V3.1",
    "zai-org/GLM-4.5",
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-ai/DeepSeek-R1",
    "moonshotai/Kimi-K2-Instruct",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "CohereLabs/command-a-reasoning-08-2025",
]

SUGGESTED_IMAGE_MODELS = [
    "Qwen/Qwen-Image",
    "black-forest-labs/FLUX.1-dev",
    "black-forest-labs/FLUX.1-Krea-dev",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "black-forest-labs/FLUX.1-schnell",
    "ByteDance/Hyper-SD",
    "UmeAiRT/FLUX.1-dev-LoRA-Modern_Pixel_art",
    "xey/sldr_flux_nsfw_v2-studio",
    "HiDream-ai/HiDream-I1-Full",
    "Kwai-Kolors/Kolors",
]

SUGGESTED_IMAGE_TO_IMAGE_MODELS = [
    "Qwen/Qwen-Image-Edit",
    "black-forest-labs/FLUX.1-Kontext-dev",
    "fofr/kontext-make-person-real",
    "jerrrycans/watermark20000",
    "fal/Pencil-Drawing-Kontext-Dev-LoRA",
]

SUGGESTED_VIDEO_MODELS = [
    "Wan-AI/Wan2.2-T2V-A14B",
    "Wan-AI/Wan2.2-TI2V-5B",
    "tencent/HunyuanVideo",
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "zai-org/CogVideoX-5b",
    "Wan-AI/Wan2.1-T2V-14B",
    "genmo/mochi-1-preview",
    "Wan-AI/Wan2.1-T2V-1.3B",
    "Lightricks/LTX-Video-0.9.7-dev",
    "Lightricks/LTX-Video-0.9.5",
    "Lightricks/LTX-Video-0.9.7-distilled",
]



# Model-specific configurations for TTS
TTS_MODEL_CONFIGS = {
    "hexgrad/Kokoro-82M": {
        "type": "kokoro",
        "supports_voice": True,
        "supports_speed": True,
        "extra_body_params": ["voice", "speed"]
    },
    "ResembleAI/chatterbox": {
        "type": "chatterbox", 
        "supports_voice": False,
        "supports_speed": False,
        "extra_body_params": ["audio_url", "exaggeration", "temperature", "cfg"]
    },
    "nari-labs/Dia-1.6B": {
        "type": "dia",
        "supports_voice": False,
        "supports_speed": False,
        "extra_body_params": []
    }
}

# -----------------------------
# Text-to-Video configuration
# -----------------------------


# Example prompts for text-to-video generation
VIDEO_EXAMPLE_PROMPTS = [
    "A young man walking on the street",
    "A corgi puppy running through a field of flowers, cinematic",
    "A futuristic city skyline at sunset with flying cars, 4k",
    "A serene beach with gentle waves and palm trees swaying",
]

# Voice options for Kokoro TTS (based on the reference app)
TTS_VOICES = {
    '🇺🇸 🚺 Heart ❤️': 'af_heart',
    '🇺🇸 🚺 Bella 🔥': 'af_bella',
    '🇺🇸 🚺 Nicole 🎧': 'af_nicole',
    '🇺🇸 🚺 Aoede': 'af_aoede',
    '🇺🇸 🚺 Kore': 'af_kore',
    '🇺🇸 🚺 Sarah': 'af_sarah',
    '🇺🇸 🚺 Nova': 'af_nova',
    '🇺🇸 🚺 Sky': 'af_sky',
    '🇺🇸 🚺 Alloy': 'af_alloy',
    '🇺🇸 🚺 Jessica': 'af_jessica',
    '🇺🇸 🚺 River': 'af_river',
    '🇺🇸 🚹 Michael': 'am_michael',
    '🇺🇸 🚹 Fenrir': 'am_fenrir',
    '🇺🇸 🚹 Puck': 'am_puck',
    '🇺🇸 🚹 Echo': 'am_echo',
    '🇺🇸 🚹 Eric': 'am_eric',
    '🇺🇸 🚹 Liam': 'am_liam',
    '🇺🇸 🚹 Onyx': 'am_onyx',
    '🇺🇸 🚹 Santa': 'am_santa',
    '🇺🇸 🚹 Adam': 'am_adam',
    '🇬🇧 🚺 Emma': 'bf_emma',
    '🇬🇧 🚺 Isabella': 'bf_isabella',
    '🇬🇧 🚺 Alice': 'bf_alice',
    '🇬🇧 🚺 Lily': 'bf_lily',
    '🇬🇧 🚹 George': 'bm_george',
    '🇬🇧 🚹 Fable': 'bm_fable',
    '🇬🇧 🚹 Lewis': 'bm_lewis',
    '🇬🇧 🚹 Daniel': 'bm_daniel',
}

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

# Example prompts for image-to-image generation
IMAGE_TO_IMAGE_EXAMPLE_PROMPTS = [
    "Turn the cat into a tiger with stripes and fierce expression",
    "Make the background a magical forest with glowing mushrooms",
    "Change the style to vintage comic book with bold colors",
    "Add a superhero cape and mask to the person",
    "Transform the building into a futuristic skyscraper",
    "Make the flowers bloom and add butterflies around them",
    "Change the weather to a stormy night with lightning",
    "Add a magical portal in the background with sparkles"
]

# Example texts for text-to-speech generation
TTS_EXAMPLE_TEXTS = [
    "Hello! Welcome to the amazing world of AI-powered text-to-speech technology.",
    "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the alphabet.",
    "In a world where technology advances at lightning speed, artificial intelligence continues to reshape our future.",
    "Imagine a world where machines can understand and respond to human emotions with perfect clarity.",
    "The future belongs to those who believe in the beauty of their dreams and have the courage to pursue them.",
    "Science is not only compatible with spirituality; it is a profound source of spirituality.",
    "The only way to do great work is to love what you do. If you haven't found it yet, keep looking.",
    "Life is what happens when you're busy making other plans. Embrace every moment with gratitude.",
    "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now."
]

# Example audio URLs for Chatterbox TTS
TTS_EXAMPLE_AUDIO_URLS = [
    "https://github.com/nazdridoy/kokoro-tts/raw/main/previews/demo.mp3",
    "https://storage.googleapis.com/chatterbox-demo-samples/prompts/male_rickmorty.mp3"
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


# -----------------------------
# OAuth / Org Access Utilities
# -----------------------------

def _parse_allowed_orgs() -> list[str]:
    """Parse comma/space separated ALLOWED_ORGS env var into a list of lowercase names."""
    raw = os.getenv("ALLOWED_ORGS", "").strip()
    if not raw:
        return []
    # support comma or whitespace separated
    parts = [p.strip().lower() for p in raw.replace("\n", ",").replace(" ", ",").split(",") if p.strip()]
    return list(dict.fromkeys(parts))  # dedupe while preserving order


def fetch_hf_identity(access_token: str) -> tuple[bool, dict | None, str]:
    """
    Call whoami-v2 to get user identity and orgs.
    Returns (success, data, error_message).
    """
    if not access_token:
        return False, None, "Missing access token"
    try:
        resp = requests.get(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
            timeout=15,
        )
        if resp.status_code != 200:
            return False, None, f"HF whoami-v2 HTTP {resp.status_code}"
        return True, resp.json(), ""
    except requests.exceptions.RequestException as e:
        return False, None, f"HF whoami-v2 error: {str(e)}"


def check_org_access(access_token: str) -> tuple[bool, str, str | None, list[str]]:
    """
    Validate that the logged-in user belongs to any of ALLOWED_ORGS.
    Returns (is_allowed, message, username, matched_orgs).
    """
    allowed_orgs = _parse_allowed_orgs()
    if not access_token:
        return False, "🔒 Please log in with Hugging Face to continue.", None, []
    if not allowed_orgs:
        return False, "❌ Access denied: ALLOWED_ORGS is not configured in Space secrets.", None, []

    ok, data, err = fetch_hf_identity(access_token)
    if not ok or not data:
        return False, f"❌ Failed to verify identity: {err}", None, []

    username = data.get("name") or data.get("fullname") or data.get("id")
    org_objs = data.get("orgs", []) or []
    user_org_names = [str(org.get("name", "")).lower() for org in org_objs if org.get("name")]
    matched = sorted(list(set(user_org_names).intersection(set(allowed_orgs))))
    if matched:
        return True, f"✅ Access granted for @{username} in org(s): {', '.join(matched)}", username, matched
    return False, f"🚫 Access denied for @{username}. Required org(s): {', '.join(allowed_orgs)}", username, []


def format_access_denied_message(message: str) -> str:
    """Return a standardized access denied message for UI display."""
    return format_error_message("Access Denied", message)


# -----------------------------
# Reasoning (<think>) utilities
# -----------------------------

def render_with_reasoning_toggle(text: str, show_reasoning: bool) -> str:
    """Render assistant text while optionally revealing content inside <think>...</think>.

    Behavior:
    - When show_reasoning is True:
      * Replace the opening <think> tag with a collapsible HTML <details> block and an opening
        fenced code block. Stream reasoning tokens inside this block as they arrive.
      * Replace the closing </think> tag with the closing fence and </details> when it appears.
    - When show_reasoning is False:
      * Remove complete <think>...</think> blocks.
      * For partial streams (no closing tag yet), trim everything from the first <think> onward.

    Safe to call on every streamed chunk; conversions are idempotent.
    """
    if not isinstance(text, str):
        return text

    # If we are NOT showing reasoning, remove it entirely. For partial streams, hide from <think> onwards.
    if not show_reasoning:
        if "<think>" not in text:
            return text
        if "</think>" not in text:
            return text.split("<think>", 1)[0]
        # Remove complete <think>...</think> blocks
        pattern_strip = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
        return pattern_strip.sub("", text)

    # Show reasoning: stream it as it arrives by converting tags into a collapsible details block
    open_block_open = "<details open><summary>Reasoning</summary>\n\n```text\n"
    open_block_closed = "<details><summary>Reasoning</summary>\n\n```text\n"
    close_block = "\n```\n</details>\n"

    # If the closing tag is not present yet, keep the block expanded while streaming
    if "</think>" not in text:
        # Replace any raw <think> with an expanded details block
        text = re.sub(r"<think>", open_block_open, text, flags=re.IGNORECASE)
        # If for any reason a closed details opening exists, switch it to open (expanded)
        text = text.replace(open_block_closed, open_block_open)
        return text

    # If the closing tag is present, render a collapsed block by default
    # 1) Ensure opening is the closed variant
    text = re.sub(r"<think>", open_block_closed, text, flags=re.IGNORECASE)
    text = text.replace(open_block_open, open_block_closed)
    # 2) Close the block
    text = re.sub(r"</think>", close_block, text, flags=re.IGNORECASE)

    return text
