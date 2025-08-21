"""
HF-Inferoxy AI Hub - Main application entry point.
A comprehensive AI platform with chat and image generation capabilities.
"""

import gradio as gr
from chat_handler import handle_chat_submit, handle_chat_retry
from image_handler import handle_image_generation, handle_image_to_image_generation
from ui_components import (
    create_main_header,
    create_chat_tab,
    create_image_tab,
    create_image_to_image_tab,
    create_footer
)
from utils import get_gradio_theme


def create_app():
    """Create and configure the main Gradio application."""
    
    # Create the main Gradio interface with tabs
    with gr.Blocks(title="HF-Inferoxy AI Hub", theme=get_gradio_theme()) as demo:
        
        # Main header
        create_main_header()
        
        with gr.Tabs() as tabs:
            
            # Chat tab
            create_chat_tab(handle_chat_submit, handle_chat_retry)
            
            # Image generation tab  
            create_image_tab(handle_image_generation)
            
            # Image-to-image tab
            create_image_to_image_tab(handle_image_to_image_generation)
        
        # Footer with helpful information
        create_footer()
    
    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch()