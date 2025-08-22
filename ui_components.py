"""
UI components for HF-Inferoxy AI Hub.
Contains functions to create different sections of the Gradio interface.
"""

import gradio as gr
from utils import (
    DEFAULT_CHAT_MODEL, DEFAULT_IMAGE_MODEL, DEFAULT_PROVIDER,
    DEFAULT_IMAGE_TO_IMAGE_MODEL,
    DEFAULT_TTS_MODEL,
    CHAT_CONFIG, IMAGE_CONFIG, IMAGE_PROVIDERS, IMAGE_MODEL_PRESETS,
    IMAGE_TO_IMAGE_MODEL_PRESETS, TTS_MODEL_PRESETS, TTS_VOICES, TTS_MODEL_CONFIGS,
    IMAGE_EXAMPLE_PROMPTS, IMAGE_TO_IMAGE_EXAMPLE_PROMPTS, TTS_EXAMPLE_TEXTS, TTS_EXAMPLE_AUDIO_URLS,
    DEFAULT_VIDEO_MODEL, VIDEO_MODEL_PRESETS, VIDEO_EXAMPLE_PROMPTS
)


def create_chat_tab(handle_chat_submit_fn, handle_chat_retry_fn=None):
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
            chat_stop = gr.Button("⏹ Stop", variant="secondary", scale=0, visible=False)
        
        # Configuration options below the chat
        with gr.Row():
            with gr.Column(scale=1):
                chat_model_name = gr.Textbox(
                    value=DEFAULT_CHAT_MODEL,
                    label="Model Name",
                    placeholder="e.g., openai/gpt-oss-20b (provider via dropdown)",
                    info="Do not include :provider in model name"
                )
                chat_provider = gr.Dropdown(
                    choices=IMAGE_PROVIDERS,
                    value=DEFAULT_PROVIDER,
                    label="Provider",
                    interactive=True
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
        
        # Connect chat events (streaming auto-detected from generator function)
        # Show stop immediately when sending
        chat_submit.click(
            fn=lambda: gr.update(visible=True),
            inputs=None,
            outputs=[chat_stop],
            queue=False
        )

        chat_send_event = chat_submit.click(
            fn=handle_chat_submit_fn,
            inputs=[chat_input, chatbot_display, chat_system_message, chat_model_name, 
                   chat_provider, chat_max_tokens, chat_temperature, chat_top_p],
            outputs=[chatbot_display, chat_input]
        )
        
        # Show stop immediately when pressing Enter
        chat_input.submit(
            fn=lambda: gr.update(visible=True),
            inputs=None,
            outputs=[chat_stop],
            queue=False
        )

        chat_enter_event = chat_input.submit(
            fn=handle_chat_submit_fn,
            inputs=[chat_input, chatbot_display, chat_system_message, chat_model_name, 
                   chat_provider, chat_max_tokens, chat_temperature, chat_top_p],
            outputs=[chatbot_display, chat_input]
        )

        # Stop current chat generation
        chat_stop.click(
            fn=lambda: gr.update(visible=False),
            inputs=None,
            outputs=[chat_stop],
            cancels=[chat_send_event, chat_enter_event],
            queue=False
        )

        # Hide stop after completion of chat events
        chat_send_event.then(lambda: gr.update(visible=False), None, [chat_stop], queue=False)
        chat_enter_event.then(lambda: gr.update(visible=False), None, [chat_stop], queue=False)

        # Enable retry icon and bind handler if provided
        if handle_chat_retry_fn is not None:
            chatbot_display.retry(
                fn=handle_chat_retry_fn,
                inputs=[chatbot_display, chat_system_message, chat_model_name, 
                        chat_provider, chat_max_tokens, chat_temperature, chat_top_p],
                outputs=chatbot_display
            )


def create_chat_tips():
    """Create the tips section for the chat tab."""
    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            ### 💡 Chat Tips
            
            **Model Format:**
            - Model only: `openai/gpt-oss-20b`
            - Select provider via the Provider dropdown (default: `auto`)
            
            **Popular Models:**
            - `openai/gpt-oss-20b` - Fast general purpose
            - `meta-llama/Llama-2-7b-chat-hf` - Chat optimized
            - `microsoft/DialoGPT-medium` - Conversation
            - `google/flan-t5-base` - Instruction following
            """)
        
        with gr.Column():
            gr.Markdown("""
            ### 🚀 Popular Providers
            
            - Select from dropdown. Default is **auto**.
            
            **Example:**
            - Model: `openai/gpt-oss-20b`, Provider: `groq`
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
                        value=DEFAULT_PROVIDER,
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
                
                # Generate and Stop buttons
                with gr.Row():
                    generate_btn = gr.Button(
                        "🎨 Generate Image", 
                        variant="primary", 
                        size="lg",
                        scale=2
                    )
                    stop_generate_btn = gr.Button("⏹ Stop", variant="secondary", visible=False)
                
                # Quick model presets
                create_image_presets(img_model_name, img_provider)
        
        # Examples for image generation
        create_image_examples(img_prompt)
        
        # Connect image generation events
        # Show stop immediately when starting generation
        generate_btn.click(
            fn=lambda: gr.update(visible=True),
            inputs=None,
            outputs=[stop_generate_btn],
            queue=False
        )

        gen_event = generate_btn.click(
            fn=handle_image_generation_fn,
            inputs=[
                img_prompt, img_model_name, img_provider, img_negative_prompt,
                img_width, img_height, img_steps, img_guidance, img_seed
            ],
            outputs=[output_image, status_text]
        )

        # Stop current image generation
        stop_generate_btn.click(
            fn=lambda: gr.update(visible=False),
            inputs=None,
            outputs=[stop_generate_btn],
            cancels=[gen_event],
            queue=False
        )

        # Hide stop after generation completes
        gen_event.then(lambda: gr.update(visible=False), None, [stop_generate_btn], queue=False)


def create_image_to_image_tab(handle_image_to_image_generation_fn):
    """
    Create the image-to-image tab interface.
    """
    with gr.Tab("🖼️ Image-to-Image", id="image-to-image"):
        with gr.Row():
            with gr.Column(scale=1):
                # Input image
                input_image = gr.Image(
                    label="Input Image", 
                    type="pil",
                    height=400,
                    show_download_button=True
                )
                
                # Model and provider inputs
                with gr.Group():
                    gr.Markdown("**🤖 Model & Provider**")
                    img2img_model_name = gr.Textbox(
                        value=DEFAULT_IMAGE_TO_IMAGE_MODEL,
                        label="Model Name",
                        placeholder="e.g., Qwen/Qwen-Image-Edit or black-forest-labs/FLUX.1-Kontext-dev"
                    )
                    img2img_provider = gr.Dropdown(
                        choices=IMAGE_PROVIDERS,
                        value=DEFAULT_PROVIDER,
                        label="Provider",
                        interactive=True
                    )
            
            with gr.Column(scale=1):
                # Output image
                output_image = gr.Image(
                    label="Generated Image", 
                    type="pil",
                    height=400,
                    show_download_button=True
                )
                status_text = gr.Textbox(
                    label="Generation Status", 
                    interactive=False,
                    lines=2
                )
            
            with gr.Column(scale=1):
                # Generation parameters
                with gr.Group():
                    gr.Markdown("**📝 Prompts**")
                    img2img_prompt = gr.Textbox(
                        value=IMAGE_TO_IMAGE_EXAMPLE_PROMPTS[0],  # Use first example as default
                        label="Prompt",
                        lines=3,
                        placeholder="Describe how you want to modify the image..."
                    )
                    img2img_negative_prompt = gr.Textbox(
                        value=IMAGE_CONFIG["negative_prompt"],
                        label="Negative Prompt",
                        lines=2,
                        placeholder="Describe what you DON'T want in the modified image..."
                    )
                
                with gr.Group():
                    gr.Markdown("**⚙️ Generation Settings**")
                    with gr.Row():
                        img2img_steps = gr.Slider(
                            minimum=10, maximum=100, value=IMAGE_CONFIG["num_inference_steps"], step=1,
                            label="Inference Steps", info="More steps = better quality"
                        )
                        img2img_guidance = gr.Slider(
                            minimum=1.0, maximum=20.0, value=IMAGE_CONFIG["guidance_scale"], step=0.5,
                            label="Guidance Scale", info="How closely to follow prompt"
                        )
                    
                    img2img_seed = gr.Slider(
                        minimum=-1, maximum=999999, value=IMAGE_CONFIG["seed"], step=1,
                        label="Seed", info="-1 for random"
                    )
                
                # Generate and Stop buttons
                with gr.Row():
                    generate_btn = gr.Button(
                        "🖼️ Generate Image-to-Image", 
                        variant="primary", 
                        size="lg",
                        scale=2
                    )
                    stop_generate_btn = gr.Button("⏹ Stop", variant="secondary", visible=False)
                
                # Quick model presets
                create_image_to_image_presets(img2img_model_name, img2img_provider)
        
        # Examples for image-to-image generation
        create_image_to_image_examples(img2img_prompt)
        
        # Connect image-to-image generation events
        # Show stop immediately when starting generation
        generate_btn.click(
            fn=lambda: gr.update(visible=True),
            inputs=None,
            outputs=[stop_generate_btn],
            queue=False
        )

        gen_event = generate_btn.click(
            fn=handle_image_to_image_generation_fn,
            inputs=[
                input_image, img2img_prompt, img2img_model_name, img2img_provider, img2img_negative_prompt,
                img2img_steps, img2img_guidance, img2img_seed
            ],
            outputs=[output_image, status_text]
        )

        # Stop current image-to-image generation
        stop_generate_btn.click(
            fn=lambda: gr.update(visible=False),
            inputs=None,
            outputs=[stop_generate_btn],
            cancels=[gen_event],
            queue=False
        )

        # Hide stop after generation completes
        gen_event.then(lambda: gr.update(visible=False), None, [stop_generate_btn], queue=False)


def create_tts_tab(handle_tts_generation_fn):
    """
    Create the text-to-speech tab interface with dynamic model-specific settings.
    """
    with gr.Tab("🎤 Text-to-Speech", id="tts"):
        with gr.Row():
            with gr.Column(scale=2):
                # Text input
                tts_text = gr.Textbox(
                    value=TTS_EXAMPLE_TEXTS[0],  # Use first example as default
                    label="Text to Convert",
                    lines=6,
                    placeholder="Enter the text you want to convert to speech..."
                )
                
                # Audio output
                output_audio = gr.Audio(
                    label="Generated Audio",
                    type="numpy",
                    interactive=False,
                    autoplay=True,
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
                    tts_model_name = gr.Dropdown(
                        choices=["hexgrad/Kokoro-82M", "ResembleAI/chatterbox", "nari-labs/Dia-1.6B"],
                        value=DEFAULT_TTS_MODEL,
                        label="Model",
                        info="Select TTS model"
                    )
                    tts_provider = gr.Dropdown(
                        choices=IMAGE_PROVIDERS,
                        value=DEFAULT_PROVIDER,
                        label="Provider",
                        interactive=True
                    )
                
                # Kokoro-specific settings (initially visible)
                with gr.Group(visible=True) as kokoro_settings:
                    gr.Markdown("**🎤 Kokoro Voice Settings**")
                    tts_voice = gr.Dropdown(
                        choices=list(TTS_VOICES.items()),
                        value="af_bella",
                        label="Voice",
                        info="Choose from various English voices"
                    )
                    tts_speed = gr.Slider(
                        minimum=0.5, maximum=2.0, value=1.0, step=0.1,
                        label="Speed", info="0.5 = slow, 2.0 = fast"
                    )
                
                # Chatterbox-specific settings (initially hidden)
                with gr.Group(visible=False) as chatterbox_settings:
                    gr.Markdown("**🎭 Chatterbox Style Settings**")
                    tts_audio_url = gr.Textbox(
                        value=TTS_EXAMPLE_AUDIO_URLS[0],
                        label="Reference Audio URL",
                        placeholder="Enter URL to reference audio file",
                        info="Audio file to match style and tone"
                    )
                    tts_exaggeration = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.25, step=0.05,
                        label="Exaggeration", info="How much to exaggerate the style"
                    )
                    tts_temperature = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.7, step=0.1,
                        label="Temperature", info="Creativity level"
                    )
                    tts_cfg = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.5, step=0.1,
                        label="CFG", info="Guidance strength"
                    )
                
                # Generate and Stop buttons
                with gr.Row():
                    generate_btn = gr.Button(
                        "🎤 Generate Speech", 
                        variant="primary", 
                        size="lg",
                        scale=2
                    )
                    stop_generate_btn = gr.Button("⏹ Stop", variant="secondary", visible=False)
                
                # Quick model presets
                create_tts_presets(tts_model_name, tts_provider)
        
        # Examples for TTS generation
        create_tts_examples(tts_text)

        # Create Chatterbox audio URL examples
        create_chatterbox_examples(tts_audio_url)
        
        # Model change handler to show/hide appropriate settings
        def on_model_change(model_name):
            if model_name == "hexgrad/Kokoro-82M":
                return gr.update(visible=True), gr.update(visible=False)
            elif model_name == "ResembleAI/chatterbox":
                return gr.update(visible=False), gr.update(visible=True)
            elif model_name == "nari-labs/Dia-1.6B":
                return gr.update(visible=False), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=False)
        
        # Connect model change event
        tts_model_name.change(
            fn=on_model_change,
            inputs=[tts_model_name],
            outputs=[kokoro_settings, chatterbox_settings]
        )
        
        # Connect TTS generation events
        # Show stop immediately when starting generation
        generate_btn.click(
            fn=lambda: gr.update(visible=True),
            inputs=None,
            outputs=[stop_generate_btn],
            queue=False
        )

        gen_event = generate_btn.click(
            fn=handle_tts_generation_fn,
            inputs=[
                tts_text, tts_model_name, tts_provider, tts_voice, tts_speed,
                tts_audio_url, tts_exaggeration, tts_temperature, tts_cfg
            ],
            outputs=[output_audio, status_text]
        )

        # Stop current TTS generation
        stop_generate_btn.click(
            fn=lambda: gr.update(visible=False),
            inputs=None,
            outputs=[stop_generate_btn],
            cancels=[gen_event],
            queue=False
        )

        # Hide stop after generation completes
        gen_event.then(lambda: gr.update(visible=False), None, [stop_generate_btn], queue=False)


def create_video_tab(handle_video_generation_fn):
    """
    Create the text-to-video tab interface.
    """
    with gr.Tab("🎬 Text-to-Video", id="video"):
        with gr.Row():
            with gr.Column(scale=2):
                # Video output
                output_video = gr.Video(
                    label="Generated Video",
                    interactive=False,
                    show_download_button=True,
                    height=480,
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
                    vid_model_name = gr.Textbox(
                        value=DEFAULT_VIDEO_MODEL,
                        label="Model Name",
                        placeholder="e.g., tencent/HunyuanVideo, Wan-AI/Wan2.2-T2V-A14B"
                    )
                    vid_provider = gr.Dropdown(
                        choices=IMAGE_PROVIDERS,
                        value=DEFAULT_PROVIDER,
                        label="Provider",
                        interactive=True
                    )

                # Generation parameters
                with gr.Group():
                    gr.Markdown("**📝 Prompt**")
                    vid_prompt = gr.Textbox(
                        value=VIDEO_EXAMPLE_PROMPTS[0],
                        label="Prompt",
                        lines=3,
                        placeholder="Describe the video you want to generate..."
                    )

                with gr.Group():
                    gr.Markdown("**⚙️ Generation Settings (optional)**")
                    with gr.Row():
                        vid_steps = gr.Slider(
                            minimum=10, maximum=100, value=20, step=1,
                            label="Inference Steps"
                        )
                        vid_guidance = gr.Slider(
                            minimum=1.0, maximum=20.0, value=7.5, step=0.5,
                            label="Guidance Scale"
                        )
                    vid_seed = gr.Slider(
                        minimum=-1, maximum=999999, value=-1, step=1,
                        label="Seed", info="-1 for random"
                    )

                # Generate and Stop buttons
                with gr.Row():
                    generate_btn = gr.Button(
                        "🎬 Generate Video",
                        variant="primary",
                        size="lg",
                        scale=2
                    )
                    stop_generate_btn = gr.Button("⏹ Stop", variant="secondary", visible=False)

                # Quick model presets
                with gr.Group():
                    gr.Markdown("**🎯 Popular Presets**")
                    for name, model, provider in VIDEO_MODEL_PRESETS:
                        btn = gr.Button(name, size="sm")
                        btn.click(
                            lambda m=model, p=provider: (m, p),
                            outputs=[vid_model_name, vid_provider]
                        )

        # Examples for video generation
        with gr.Group():
            gr.Markdown("**🌟 Example Prompts**")
            gr.Examples(
                examples=[[prompt] for prompt in VIDEO_EXAMPLE_PROMPTS],
                inputs=vid_prompt
            )

        # Connect video generation events
        generate_btn.click(
            fn=lambda: gr.update(visible=True),
            inputs=None,
            outputs=[stop_generate_btn],
            queue=False
        )

        gen_event = generate_btn.click(
            fn=handle_video_generation_fn,
            inputs=[
                vid_prompt, vid_model_name, vid_provider,
                vid_steps, vid_guidance, vid_seed
            ],
            outputs=[output_video, status_text]
        )

        # Stop current video generation
        stop_generate_btn.click(
            fn=lambda: gr.update(visible=False),
            inputs=None,
            outputs=[stop_generate_btn],
            cancels=[gen_event],
            queue=False
        )

        # Hide stop after generation completes
        gen_event.then(lambda: gr.update(visible=False), None, [stop_generate_btn], queue=False)


def create_image_to_image_presets(img2img_model_name, img2img_provider):
    """Create quick model presets for image-to-image generation."""
    with gr.Group():
        gr.Markdown("**🎯 Popular Presets**")
        
        for name, model, provider in IMAGE_TO_IMAGE_MODEL_PRESETS:
            btn = gr.Button(name, size="sm")
            btn.click(
                lambda m=model, p=provider: (m, p),
                outputs=[img2img_model_name, img2img_provider]
            )


def create_image_to_image_examples(img2img_prompt):
    """Create example prompts for image-to-image generation."""
    with gr.Group():
        gr.Markdown("**🌟 Example Prompts**")
        img2img_examples = gr.Examples(
            examples=[[prompt] for prompt in IMAGE_TO_IMAGE_EXAMPLE_PROMPTS],
            inputs=img2img_prompt
        )


def create_tts_presets(tts_model_name, tts_provider):
    """Create quick model presets for text-to-speech generation."""
    with gr.Group():
        gr.Markdown("**🎯 Popular Presets**")
        
        for name, model, provider in TTS_MODEL_PRESETS:
            btn = gr.Button(name, size="sm")
            btn.click(
                lambda m=model, p=provider: (m, p),
                outputs=[tts_model_name, tts_provider]
            )


def create_tts_examples(tts_text):
    """Create example texts for text-to-speech generation."""
    with gr.Group():
        gr.Markdown("**🌟 Example Texts**")
        tts_examples = gr.Examples(
            examples=[[text] for text in TTS_EXAMPLE_TEXTS],
            inputs=tts_text
        )


def create_chatterbox_examples(tts_audio_url):
    """Create example audio URLs for Chatterbox TTS."""
    with gr.Group():
        gr.Markdown("**🎵 Example Reference Audio URLs**")
        chatterbox_examples = gr.Examples(
            examples=[[url] for url in TTS_EXAMPLE_AUDIO_URLS],
            inputs=tts_audio_url
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
    
    A comprehensive AI platform combining chat, image generation, image-to-image, text-to-video, and text-to-speech capabilities with intelligent token management through HF-Inferoxy.
    
    **Features:**
    - 💬 **Smart Chat**: Conversational AI with streaming responses
    - 🎨 **Image Generation**: Text-to-image creation with multiple providers  
    - 🖼️ **Image-to-Image**: Transform and modify existing images with AI
    - 🎬 **Text-to-Video**: Generate short videos from text prompts
    - 🎤 **Text-to-Speech**: Convert text to natural-sounding speech
    - 🔄 **Intelligent Token Management**: Automatic token rotation and error handling
    - 🌐 **Multi-Provider Support**: Works with HF Inference, Cerebras, Cohere, Groq, Together, Fal.ai, and more
    """)


def create_footer():
    """Create the footer with helpful information."""
    gr.Markdown("""
    ---
    ### 📚 Quick Guide
    
    **Chat**
    - Type a message, pick a model, set provider (defaults to `auto`).
    - Tune system message, temperature, top‑p, and max new tokens.
    
    **Image**
    - Write a prompt and select model/provider.
    - Optional: width/height (÷8), steps, guidance, seed, negative prompt.
    
    **Image‑to‑Image**
    - Upload an input image and describe the change you want.
    - Use the same settings as Image for quality and control.
    
    **Video**
    - Provide a short motion prompt; optionally set steps, guidance, and seed.
    
    **Text‑to‑Speech**
    - Enter text, choose a TTS model, and adjust voice/style if available.
    
    **Providers**
    - Default is `auto`. You can choose from providers available via your HF‑Inferoxy setup.
    
    **Docs**
    - HF‑Inferoxy: https://nazdridoy.github.io/hf-inferoxy/
    
    Built with HF‑Inferoxy for secure token management and provider routing.
    """)


