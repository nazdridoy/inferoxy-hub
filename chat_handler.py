"""
Chat functionality handler for AI-Inferoxy AI Hub.
Handles chat completion requests with streaming responses.
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
    validate_proxy_key, 
    format_error_message,
    render_with_reasoning_toggle
)

# Timeout configuration for inference requests
INFERENCE_TIMEOUT = 120  # 2 minutes max for inference


def chat_respond(
    message,
    history: list[dict[str, str]],
    system_message,
    model_name,
    provider_override,
    max_tokens,
    temperature,
    top_p,
    client_name: str | None = None,
):
    """
    Chat completion function using AI-Inferoxy token management.
    """
    # Validate proxy API key
    is_valid, error_msg = validate_proxy_key()
    if not is_valid:
        yield error_msg
        return
    
    proxy_api_key = os.getenv("PROXY_KEY")
    
    token_id = None
    try:
        # Get token from AI-Inferoxy proxy server with timeout handling
        print(f"🔑 Chat: Requesting token from proxy...")
        token, token_id = get_proxy_token(api_key=proxy_api_key)
        print(f"✅ Chat: Got token: {token_id}")
        
        # Enforce explicit provider selection via dropdown
        model = model_name
        provider = provider_override or "auto"
        
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
        
        print(f"🚀 Chat: Client created, starting inference with timeout...")
        
        chat_completion_kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
            "temperature": temperature,
            "top_p": top_p,
        }

        response = ""
        
        print(f"📡 Chat: Making streaming request with {INFERENCE_TIMEOUT}s timeout...")
        
        # Create streaming function for timeout handling
        def create_stream():
            return client.chat_completion(**chat_completion_kwargs)
        
        # Execute with timeout using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(create_stream)
            
            try:
                # Get the stream with timeout
                stream = future.result(timeout=INFERENCE_TIMEOUT)
                print(f"🔄 Chat: Got stream, starting to iterate...")

                # Track streaming time to detect hangs
                last_token_time = time.time()
                token_timeout = 30  # 30 seconds between tokens
                
                for message in stream:
                    current_time = time.time()
                    
                    # Check if we've been waiting too long for a token
                    if current_time - last_token_time > token_timeout:
                        raise TimeoutError(f"No response received for {token_timeout} seconds during streaming")
                    
                    choices = message.choices
                    token_content = ""
                    if len(choices) and choices[0].delta.content:
                        token_content = choices[0].delta.content
                        last_token_time = current_time  # Reset timer when we get content

                    response += token_content
                    yield response
                    
            except FutureTimeoutError:
                future.cancel()  # Cancel the running task
                raise TimeoutError(f"Chat request timed out after {INFERENCE_TIMEOUT} seconds")
        
        # Report successful token usage
        if token_id:
            report_token_status(token_id, "success", api_key=proxy_api_key, client_name=client_name)
            
    except ConnectionError as e:
        # Handle proxy connection errors
        error_msg = f"Cannot connect to AI-Inferoxy server: {str(e)}"
        print(f"🔌 Chat connection error: {error_msg}")
        if token_id:
            report_token_status(token_id, "error", error_msg, api_key=proxy_api_key, client_name=client_name)
        yield format_error_message("Connection Error", "Unable to connect to the proxy server. Please check if it's running.")
        
    except TimeoutError as e:
        # Handle timeout errors
        error_msg = f"Request timed out: {str(e)}"
        print(f"⏰ Chat timeout: {error_msg}")
        if token_id:
            report_token_status(token_id, "error", error_msg, api_key=proxy_api_key, client_name=client_name)
        yield format_error_message("Timeout Error", "The request took too long. The server may be overloaded. Please try again.")
        
    except HfHubHTTPError as e:
        # Handle HuggingFace API errors
        error_msg = str(e)
        print(f"🤗 Chat HF error: {error_msg}")
        if token_id:
            report_token_status(token_id, "error", error_msg, api_key=proxy_api_key, client_name=client_name)
        
        # Provide more user-friendly error messages
        if "401" in error_msg:
            yield format_error_message("Authentication Error", "Invalid or expired API token. The proxy will provide a new token on retry.")
        elif "402" in error_msg:
            yield format_error_message("Quota Exceeded", "API quota exceeded. The proxy will try alternative providers.")
        elif "429" in error_msg:
            yield format_error_message("Rate Limited", "Too many requests. Please wait a moment and try again.")
        else:
            yield format_error_message("HuggingFace API Error", error_msg)
        
    except Exception as e:
        # Handle all other errors
        error_msg = str(e)
        print(f"❌ Chat unexpected error: {error_msg}")
        if token_id:
            report_token_status(token_id, "error", error_msg, api_key=proxy_api_key)
        yield format_error_message("Unexpected Error", f"An unexpected error occurred: {error_msg}")


def handle_chat_submit(message, history, system_msg, model_name, provider, max_tokens, temperature, top_p, hf_token: gr.OAuthToken = None):
    """
    Handle chat submission and manage conversation history with streaming.
    """
    if not message.strip():
        yield history, ""
        return

    # Require sign-in: if no token present, prompt login
    access_token = getattr(hf_token, "token", None) if hf_token is not None else None
    username = None
    if not access_token:
        assistant_response = format_error_message("Access Required", "Please sign in with Hugging Face (sidebar Login button).")
        current_history = history + [{"role": "assistant", "content": assistant_response}]
        yield current_history, ""
        return
    
    # Add user message to history
    history = history + [{"role": "user", "content": message}]
    
    # Generate response with streaming
    response_generator = chat_respond(
        message, 
        history[:-1],  # Don't include the current message in history for the function
        system_msg, 
        model_name,
        provider,
        max_tokens, 
        temperature, 
        top_p,
        client_name=username
    )
    
    # Stream the assistant response token by token
    assistant_response = ""
    for partial_response in response_generator:
        assistant_response = render_with_reasoning_toggle(partial_response, True)
        # Update history with the current partial response and yield it
        current_history = history + [{"role": "assistant", "content": assistant_response}]
        yield current_history, ""


def handle_chat_retry(history, system_msg, model_name, provider, max_tokens, temperature, top_p, hf_token: gr.OAuthToken = None, retry_data=None):
    """
    Retry the assistant response for the selected message.
    Works with gr.Chatbot.retry() which provides retry_data.index for the message.
    """
    # Require sign-in: if no token present, prompt login
    access_token = getattr(hf_token, "token", None) if hf_token is not None else None
    username = None
    if not access_token:
        assistant_response = format_error_message("Access Required", "Please sign in with Hugging Face (sidebar Login button).")
        current_history = (history or []) + [{"role": "assistant", "content": assistant_response}]
        yield current_history
        return
    # Guard: empty history
    if not history:
        yield history
        return

    # Determine which assistant message index to retry
    retry_index = None
    try:
        retry_index = getattr(retry_data, "index", None)
    except Exception:
        retry_index = None

    if retry_index is None:
        # Fallback to last assistant message
        retry_index = len(history) - 1

    # Trim history up to the message being retried (exclude that assistant msg)
    trimmed_history = list(history[:retry_index])

    # Find the most recent user message before retry_index
    last_user_idx = None
    for idx in range(retry_index - 1, -1, -1):
        if trimmed_history[idx].get("role") == "user":
            last_user_idx = idx
            break

    # Nothing to retry if no prior user message
    if last_user_idx is None:
        yield history
        return

    # Message to retry and prior conversation context (before that user msg)
    message = trimmed_history[last_user_idx].get("content", "")
    prior_history = trimmed_history[:last_user_idx]

    if not message.strip():
        yield history
        return

    # Stream a new assistant response
    response_generator = chat_respond(
        message,
        prior_history,
        system_msg,
        model_name,
        provider,
        max_tokens,
        temperature,
        top_p,
        client_name=username
    )

    assistant_response = ""
    for partial_response in response_generator:
        assistant_response = render_with_reasoning_toggle(partial_response, True)
        current_history = trimmed_history + [{"role": "assistant", "content": assistant_response}]
        yield current_history
