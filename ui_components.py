"""
UI components for HF-Inferoxy AI Hub.
Contains functions to create different sections of the Gradio interface.
"""

import gradio as gr
from utils import (
    DEFAULT_CHAT_MODEL, DEFAULT_IMAGE_MODEL, DEFAULT_IMAGE_PROVIDER,
    CHAT_CONFIG, IMAGE_CONFIG, IMAGE_PROVIDERS, IMAGE_MODEL_PRESETS,
    IMAGE_EXAMPLE_PROMPTS
)


def create_chat_tab(handle_chat_submit_fn):
    """
    Create the chat tab interface.
    """
    with gr.Tab("💬 Chat Assistant", id="chat"):
        # Chat interface at the top - most prominent
        chatbot_display = gr.Chatbot(
            label="Chat",
            type="messages",
            height=800,
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
                    value=DEFAULT_CHAT_MODEL,
                    label="Model Name",
                    placeholder="e.g., openai/gpt-oss-20b or openai/gpt-oss-20b:fireworks-ai"
                )
                chat_system_message = gr.Textbox(
                    value=CHAT_CONFIG["system_message"],
                    label="System Message",
                    lines=2,
                    placeholder="Define the assistant's personality and behavior..."
                )
            
            with gr.Column(scale=1):
                chat_max_tokens = gr.Slider(
                    minimum=1, maximum=4096, value=CHAT_CONFIG["max_tokens"], step=1,
                    label="Max New Tokens"
                )
                chat_temperature = gr.Slider(
                    minimum=0.1, maximum=2.0, value=CHAT_CONFIG["temperature"], step=0.1,
                    label="Temperature"
                )
                chat_top_p = gr.Slider(
                    minimum=0.1, maximum=1.0, value=CHAT_CONFIG["top_p"], step=0.05,
                    label="Top-p (nucleus sampling)"
                )
        
        # Configuration tips below the chat
        create_chat_tips()
        
        # Connect chat events with streaming enabled
        chat_submit.click(
            fn=handle_chat_submit_fn,
            inputs=[chat_input, chatbot_display, chat_system_message, chat_model_name, 
                   chat_max_tokens, chat_temperature, chat_top_p],
            outputs=[chatbot_display, chat_input],
            stream=True
        )
        
        chat_input.submit(
            fn=handle_chat_submit_fn,
            inputs=[chat_input, chatbot_display, chat_system_message, chat_model_name, 
                   chat_max_tokens, chat_temperature, chat_top_p],
            outputs=[chatbot_display, chat_input],
            stream=True
        )


def create_chat_tips():
    """Create the tips section for the chat tab."""
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


def create_image_tab(handle_image_generation_fn):
    """
    Create the image generation tab interface.
    """
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
                        value=DEFAULT_IMAGE_MODEL,
                        label="Model Name",
                        placeholder="e.g., Qwen/Qwen-Image or stabilityai/stable-diffusion-xl-base-1.0"
                    )
                    img_provider = gr.Dropdown(
                        choices=IMAGE_PROVIDERS,
                        value=DEFAULT_IMAGE_PROVIDER,
                        label="Provider",
                        interactive=True
                    )
                
                # Generation parameters
                with gr.Group():
                    gr.Markdown("**📝 Prompts**")
                    img_prompt = gr.Textbox(
                        value=IMAGE_EXAMPLE_PROMPTS[0],  # Use first example as default
                        label="Prompt",
                        lines=3,
                        placeholder="Describe the image you want to generate..."
                    )
                    img_negative_prompt = gr.Textbox(
                        value=IMAGE_CONFIG["negative_prompt"],
                        label="Negative Prompt",
                        lines=2,
                        placeholder="Describe what you DON'T want in the image..."
                    )
                
                with gr.Group():
                    gr.Markdown("**⚙️ Generation Settings**")
                    with gr.Row():
                        img_width = gr.Slider(
                            minimum=256, maximum=2048, value=IMAGE_CONFIG["width"], step=64,
                            label="Width", info="Must be divisible by 8"
                        )
                        img_height = gr.Slider(
                            minimum=256, maximum=2048, value=IMAGE_CONFIG["height"], step=64,
                            label="Height", info="Must be divisible by 8"
                        )
                    
                    with gr.Row():
                        img_steps = gr.Slider(
                            minimum=10, maximum=100, value=IMAGE_CONFIG["num_inference_steps"], step=1,
                            label="Inference Steps", info="More steps = better quality"
                        )
                        img_guidance = gr.Slider(
                            minimum=1.0, maximum=20.0, value=IMAGE_CONFIG["guidance_scale"], step=0.5,
                            label="Guidance Scale", info="How closely to follow prompt"
                        )
                    
                    img_seed = gr.Slider(
                        minimum=-1, maximum=999999, value=IMAGE_CONFIG["seed"], step=1,
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
                create_image_presets(img_model_name, img_provider)
        
        # Examples for image generation
        create_image_examples(img_prompt)
        
        # Connect image generation events
        generate_btn.click(
            fn=handle_image_generation_fn,
            inputs=[
                img_prompt, img_model_name, img_provider, img_negative_prompt,
                img_width, img_height, img_steps, img_guidance, img_seed
            ],
            outputs=[output_image, status_text]
        )


def create_image_presets(img_model_name, img_provider):
    """Create quick model presets for image generation."""
    with gr.Group():
        gr.Markdown("**🎯 Popular Presets**")
        
        for name, model, provider in IMAGE_MODEL_PRESETS:
            btn = gr.Button(name, size="sm")
            btn.click(
                lambda m=model, p=provider: (m, p),
                outputs=[img_model_name, img_provider]
            )


def create_image_examples(img_prompt):
    """Create example prompts for image generation."""
    with gr.Group():
        gr.Markdown("**🌟 Example Prompts**")
        img_examples = gr.Examples(
            examples=[[prompt] for prompt in IMAGE_EXAMPLE_PROMPTS],
            inputs=img_prompt
        )


def create_main_header():
    """Create the main header for the application."""
    gr.Markdown("""
    # 🚀 HF-Inferoxy AI Hub
    
    A comprehensive AI platform combining chat and image generation capabilities with intelligent token management through HF-Inferoxy.
    
    **Features:**
    - 💬 **Smart Chat**: Conversational AI with streaming responses
    - 🎨 **Image Generation**: Text-to-image creation with multiple providers  
    - 🔄 **Intelligent Token Management**: Automatic token rotation and error handling
    - 🌐 **Multi-Provider Support**: Works with HF Inference, Cerebras, Cohere, Groq, Together, Fal.ai, and more
    """)


def create_footer():
    """Create the footer with helpful information."""
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
    - **fal-ai**: High-quality image generation (default for images)
    - **hf-inference**: Core API with comprehensive model support
    - **cerebras**: High-performance inference 
    - **cohere**: Advanced language models with multilingual support
    - **groq**: Ultra-fast inference, optimized for speed
    - **together**: Collaborative AI hosting, wide model support
    - **nebius**: Cloud-native services with enterprise features
    - **nscale**: Optimized inference performance 
    - **replicate**: Collaborative AI hosting
    
    **Built with ❤️ using [HF-Inferoxy](https://nazdridoy.github.io/hf-inferoxy/) for intelligent token management**
    """)
