"""
Chat functionality handler for HF-Inferoxy AI Hub.
Handles chat completion requests with streaming responses.
"""

import os
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from hf_token_utils import get_proxy_token, report_token_status
from utils import (
    validate_proxy_key, 
    parse_model_and_provider, 
    format_error_message
)


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
    # Validate proxy API key
    is_valid, error_msg = validate_proxy_key()
    if not is_valid:
        yield error_msg
        return
    
    proxy_api_key = os.getenv("PROXY_KEY")
    
    try:
        # Get token from HF-Inferoxy proxy server
        print(f"🔑 Chat: Requesting token from proxy...")
        token, token_id = get_proxy_token(api_key=proxy_api_key)
        print(f"✅ Chat: Got token: {token_id}")
        
        # Parse model name and provider if specified
        model, provider = parse_model_and_provider(model_name)
        
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
        yield format_error_message("HuggingFace API Error", str(e))
        
    except Exception as e:
        # Report other errors
        if 'token_id' in locals():
            report_token_status(token_id, "error", str(e), api_key=proxy_api_key)
        yield format_error_message("Unexpected Error", str(e))


def handle_chat_submit(message, history, system_msg, model_name, max_tokens, temperature, top_p):
    """
    Handle chat submission and manage conversation history.
    """
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
