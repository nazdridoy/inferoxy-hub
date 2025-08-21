import gradio as gr
import os
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from hf_token_utils import get_proxy_token, report_token_status
import PIL.Image
import io


def chat_respond(
    message,
    history: list[dict[str, str]],
    system_message,
    model_name,
    max_tokens,
    temperature,
    top_p,
):
    """
    Chat completion function using HF-Inferoxy token management.
    """
    # Get proxy API key from environment variable (set in HuggingFace Space secrets)
    proxy_api_key = os.getenv("PROXY_KEY")
    if not proxy_api_key:
        yield "❌ Error: PROXY_KEY not found in environment variables. Please set it in your HuggingFace Space secrets."
        return
    
    try:
        # Get token from HF-Inferoxy proxy server
        print(f"🔑 Chat: Requesting token from proxy...")
        token, token_id = get_proxy_token(api_key=proxy_api_key)
        print(f"✅ Chat: Got token: {token_id}")
        
        # Parse model name and provider if specified
        if ":" in model_name:
            model, provider = model_name.split(":", 1)
        else:
            model = model_name
            provider = None
        
        print(f"🤖 Chat: Using model='{model}', provider='{provider if provider else 'auto'}'")
        
        # Prepare messages first
        messages = [{"role": "system", "content": system_message}]
        messages.extend(history)
        messages.append({"role": "user", "content": message})

        print(f"💬 Chat: Prepared {len(messages)} messages, creating client...")
        
        # Create client with provider (auto if none specified) and always pass model
        client = InferenceClient(
            provider=provider if provider else "auto", 
            api_key=token
        )
        
        print(f"🚀 Chat: Client created, starting inference...")
        
        chat_completion_kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
            "temperature": temperature,
            "top_p": top_p,
        }

        response = ""
        
        print(f"📡 Chat: Making streaming request...")
        stream = client.chat_completion(**chat_completion_kwargs)
        print(f"🔄 Chat: Got stream, starting to iterate...")

        for message in stream:
            choices = message.choices
            token_content = ""
            if len(choices) and choices[0].delta.content:
                token_content = choices[0].delta.content

            response += token_content
            yield response
        
        # Report successful token usage
        report_token_status(token_id, "success", api_key=proxy_api_key)
        
    except HfHubHTTPError as e:
        # Report HF Hub errors
        if 'token_id' in locals():
            report_token_status(token_id, "error", str(e), api_key=proxy_api_key)
        yield f"❌ HuggingFace API Error: {str(e)}"
        
    except Exception as e:
        # Report other errors
        if 'token_id' in locals():
            report_token_status(token_id, "error", str(e), api_key=proxy_api_key)
        yield f"❌ Unexpected Error: {str(e)}"


def generate_image(
    prompt: str,
    model_name: str,
    provider: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    seed: int = -1,
):
    """
    Generate an image using the specified model and provider through HF-Inferoxy.
    """
    # Get proxy API key from environment variable (set in HuggingFace Space secrets)
    proxy_api_key = os.getenv("PROXY_KEY")
    if not proxy_api_key:
        return None, "❌ Error: PROXY_KEY not found in environment variables. Please set it in your HuggingFace Space secrets."
    
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
        
        return image, f"✅ Image generated successfully using {model_name} on {provider}!"
        
    except HfHubHTTPError as e:
        # Report HF Hub errors
        if 'token_id' in locals():
            report_token_status(token_id, "error", str(e), api_key=proxy_api_key)
        return None, f"❌ HuggingFace API Error: {str(e)}"
        
    except Exception as e:
        # Report other errors
        if 'token_id' in locals():
            report_token_status(token_id, "error", str(e), api_key=proxy_api_key)
        return None, f"❌ Unexpected Error: {str(e)}"


def validate_dimensions(width, height):
    """Validate that dimensions are divisible by 8 (required by most diffusion models)"""
    if width % 8 != 0 or height % 8 != 0:
        return False, "Width and height must be divisible by 8"
    return True, ""


# Create the main Gradio interface with tabs
with gr.Blocks(title="HF-Inferoxy AI Hub", theme=gr.themes.Soft()) as demo:
    
    # Main header
    gr.Markdown("""
    # 🚀 HF-Inferoxy AI Hub
    
    A comprehensive AI platform combining chat and image generation capabilities with intelligent token management through HF-Inferoxy.
    
    **Features:**
    - 💬 **Smart Chat**: Conversational AI with streaming responses
    - 🎨 **Image Generation**: Text-to-image creation with multiple providers  
    - 🔄 **Intelligent Token Management**: Automatic token rotation and error handling
    - 🌐 **Multi-Provider Support**: Works with HF Inference, Cerebras, Cohere, Groq, Together, Fal.ai, and more
    """)
    
    with gr.Tabs() as tabs:
        
        # ==================== CHAT TAB ====================
        with gr.Tab("💬 Chat Assistant", id="chat"):
            # Chat interface at the top - most prominent
            chatbot_display = gr.Chatbot(
                label="Chat",
                type="messages",
                height=500,
                show_copy_button=True
            )
            
            # Chat input
            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="Type your message here...",
                    label="Message",
                    scale=4,
                    container=False
                )
                chat_submit = gr.Button("Send", variant="primary", scale=1)
            
            # Configuration options below the chat
            with gr.Row():
                with gr.Column(scale=1):
                    chat_model_name = gr.Textbox(
                        value="openai/gpt-oss-20b",
                        label="Model Name",
                        placeholder="e.g., openai/gpt-oss-20b or openai/gpt-oss-20b:fireworks-ai"
                    )
                    chat_system_message = gr.Textbox(
                        value="You are a helpful and friendly AI assistant. Provide clear, accurate, and helpful responses.",
                        label="System Message",
                        lines=2,
                        placeholder="Define the assistant's personality and behavior..."
                    )
                
                with gr.Column(scale=1):
                    chat_max_tokens = gr.Slider(
                        minimum=1, maximum=4096, value=1024, step=1,
                        label="Max New Tokens"
                    )
                    chat_temperature = gr.Slider(
                        minimum=0.1, maximum=2.0, value=0.7, step=0.1,
                        label="Temperature"
                    )
                    chat_top_p = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.95, step=0.05,
                        label="Top-p (nucleus sampling)"
                    )
            
            # Configuration tips below the chat
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    ### 💡 Chat Tips
                    
                    **Model Format:**
                    - Single model: `openai/gpt-oss-20b` (uses auto provider)
                    - With provider: `openai/gpt-oss-20b:fireworks-ai`
                    
                    **Popular Models:**
                    - `openai/gpt-oss-20b` - Fast general purpose
                    - `meta-llama/Llama-2-7b-chat-hf` - Chat optimized
                    - `microsoft/DialoGPT-medium` - Conversation
                    - `google/flan-t5-base` - Instruction following
                    """)
                
                with gr.Column():
                    gr.Markdown("""
                    ### 🚀 Popular Providers
                    
                    - **auto** - Let HF choose best provider (default)
                    - **fireworks-ai** - Fast and reliable
                    - **cerebras** - High performance
                    - **groq** - Ultra-fast inference  
                    - **together** - Wide model support
                    - **cohere** - Advanced language models
                    
                    **Examples:**
                    - `openai/gpt-oss-20b` (auto provider)
                    - `openai/gpt-oss-20b:fireworks-ai` (specific provider)
                    """)
            
            # Chat functionality
            def handle_chat_submit(message, history, system_msg, model_name, max_tokens, temperature, top_p):
                if not message.strip():
                    return history, ""
                
                # Add user message to history
                history = history + [{"role": "user", "content": message}]
                
                # Generate response
                response_generator = chat_respond(
                    message, 
                    history[:-1],  # Don't include the current message in history for the function
                    system_msg, 
                    model_name, 
                    max_tokens, 
                    temperature, 
                    top_p
                )
                
                # Get the final response
                assistant_response = ""
                for partial_response in response_generator:
                    assistant_response = partial_response
                
                # Add assistant response to history
                history = history + [{"role": "assistant", "content": assistant_response}]
                
                return history, ""
            
            # Connect chat events
            chat_submit.click(
                fn=handle_chat_submit,
                inputs=[chat_input, chatbot_display, chat_system_message, chat_model_name, 
                       chat_max_tokens, chat_temperature, chat_top_p],
                outputs=[chatbot_display, chat_input]
            )
            
            chat_input.submit(
                fn=handle_chat_submit,
                inputs=[chat_input, chatbot_display, chat_system_message, chat_model_name, 
                       chat_max_tokens, chat_temperature, chat_top_p],
                outputs=[chatbot_display, chat_input]
            )
        
        # ==================== IMAGE GENERATION TAB ====================
        with gr.Tab("🎨 Image Generator", id="image"):
            with gr.Row():
                with gr.Column(scale=2):
                    # Image output
                    output_image = gr.Image(
                        label="Generated Image", 
                        type="pil",
                        height=600,
                        show_download_button=True
                    )
                    status_text = gr.Textbox(
                        label="Generation Status", 
                        interactive=False,
                        lines=2
                    )
                
                with gr.Column(scale=1):
                    # Model and provider inputs
                    with gr.Group():
                        gr.Markdown("**🤖 Model & Provider**")
                        img_model_name = gr.Textbox(
                            value="stabilityai/stable-diffusion-xl-base-1.0",
                            label="Model Name",
                            placeholder="e.g., stabilityai/stable-diffusion-xl-base-1.0"
                        )
                        img_provider = gr.Dropdown(
                            choices=["hf-inference", "fal-ai", "nebius", "nscale", "replicate", "together"],
                            value="hf-inference",
                            label="Provider",
                            interactive=True
                        )
                    
                    # Generation parameters
                    with gr.Group():
                        gr.Markdown("**📝 Prompts**")
                        img_prompt = gr.Textbox(
                            value="A beautiful landscape with mountains and a lake at sunset, photorealistic, 8k, highly detailed",
                            label="Prompt",
                            lines=3,
                            placeholder="Describe the image you want to generate..."
                        )
                        img_negative_prompt = gr.Textbox(
                            value="blurry, low quality, distorted, deformed, ugly, bad anatomy",
                            label="Negative Prompt",
                            lines=2,
                            placeholder="Describe what you DON'T want in the image..."
                        )
                    
                    with gr.Group():
                        gr.Markdown("**⚙️ Generation Settings**")
                        with gr.Row():
                            img_width = gr.Slider(
                                minimum=256, maximum=2048, value=1024, step=64,
                                label="Width", info="Must be divisible by 8"
                            )
                            img_height = gr.Slider(
                                minimum=256, maximum=2048, value=1024, step=64,
                                label="Height", info="Must be divisible by 8"
                            )
                        
                        with gr.Row():
                            img_steps = gr.Slider(
                                minimum=10, maximum=100, value=20, step=1,
                                label="Inference Steps", info="More steps = better quality"
                            )
                            img_guidance = gr.Slider(
                                minimum=1.0, maximum=20.0, value=7.5, step=0.5,
                                label="Guidance Scale", info="How closely to follow prompt"
                            )
                        
                        img_seed = gr.Slider(
                            minimum=-1, maximum=999999, value=-1, step=1,
                            label="Seed", info="-1 for random"
                        )
                    
                    # Generate button
                    generate_btn = gr.Button(
                        "🎨 Generate Image", 
                        variant="primary", 
                        size="lg",
                        scale=2
                    )
                    
                    # Quick model presets
                    with gr.Group():
                        gr.Markdown("**🎯 Popular Presets**")
                        preset_buttons = []
                        presets = [
                            ("SDXL (HF)", "stabilityai/stable-diffusion-xl-base-1.0", "hf-inference"),
                            ("FLUX.1 (Nebius)", "black-forest-labs/FLUX.1-dev", "nebius"), 
                            ("Qwen (Fal.ai)", "Qwen/Qwen-Image", "fal-ai"),
                            ("SDXL (NScale)", "stabilityai/stable-diffusion-xl-base-1.0", "nscale"),
                        ]
                        
                        for name, model, provider in presets:
                            btn = gr.Button(name, size="sm")
                            btn.click(
                                lambda m=model, p=provider: (m, p),
                                outputs=[img_model_name, img_provider]
                            )
            
            # Examples for image generation
            with gr.Group():
                gr.Markdown("**🌟 Example Prompts**")
                img_examples = gr.Examples(
                    examples=[
                        ["A majestic dragon flying over a medieval castle, epic fantasy art, detailed, 8k"],
                        ["A serene Japanese garden with cherry blossoms, zen atmosphere, peaceful, high quality"],
                        ["A futuristic cityscape with flying cars and neon lights, cyberpunk style, cinematic"],
                        ["A cute robot cat playing with yarn, adorable, cartoon style, vibrant colors"],
                        ["A magical forest with glowing mushrooms and fairy lights, fantasy, ethereal beauty"],
                        ["Portrait of a wise old wizard with flowing robes, magical aura, fantasy character art"],
                        ["A cozy coffee shop on a rainy day, warm lighting, peaceful atmosphere, detailed"],
                        ["An astronaut floating in space with Earth in background, photorealistic, stunning"]
                    ],
                    inputs=img_prompt
                )
    
    # Event handlers for image generation
    def on_generate_image(prompt_val, model_val, provider_val, negative_prompt_val, width_val, height_val, steps_val, guidance_val, seed_val):
        # Validate dimensions
        is_valid, error_msg = validate_dimensions(width_val, height_val)
        if not is_valid:
            return None, f"❌ Validation Error: {error_msg}"
        
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
    
    # Connect image generation events
    generate_btn.click(
        fn=on_generate_image,
        inputs=[
            img_prompt, img_model_name, img_provider, img_negative_prompt,
            img_width, img_height, img_steps, img_guidance, img_seed
        ],
        outputs=[output_image, status_text]
    )
    
    # Footer with helpful information
    gr.Markdown("""
    ---
    ### 📚 How to Use
    
    **Chat Tab:**
    - Enter your message and customize the AI's behavior with system messages
    - Choose models and providers using the format `model:provider` 
    - Adjust temperature for creativity and top-p for response diversity
    
    **Image Tab:**
    - Write detailed prompts describing your desired image
    - Use negative prompts to avoid unwanted elements  
    - Experiment with different models and providers for varied styles
    - Higher inference steps = better quality but slower generation
    
    **Supported Providers:**
    - **hf-inference**: Core API with comprehensive model support
    - **cerebras**: High-performance inference 
    - **cohere**: Advanced language models with multilingual support
    - **groq**: Ultra-fast inference, optimized for speed
    - **together**: Collaborative AI hosting, wide model support
    - **fal-ai**: High-quality image generation
    - **nebius**: Cloud-native services with enterprise features
    - **nscale**: Optimized inference performance 
    - **replicate**: Collaborative AI hosting
    
    **Built with ❤️ using [HF-Inferoxy](https://nazdridoy.github.io/hf-inferoxy/) for intelligent token management**
    """)


if __name__ == "__main__":
    demo.launch()
