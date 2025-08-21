# hf_token_utils.py
import os
import requests
import json
import time
from typing import Dict, Optional, Any, Tuple
from requests.exceptions import ConnectionError, Timeout, RequestException
from utils import get_proxy_url, validate_proxy_url

# Timeout and retry configuration
REQUEST_TIMEOUT = 30  # 30 seconds timeout
RETRY_ATTEMPTS = 2
RETRY_DELAY = 1  # 1 second delay between retries

def get_proxy_token(proxy_url: str = None, api_key: str = None) -> Tuple[str, str]:
    """
    Get a valid token from the proxy server with timeout and retry logic.
    
    Args:
        proxy_url: URL of the HF-Inferoxy server (optional, will use PROXY_URL env var if not provided)
        api_key: Your API key for authenticating with the proxy server
        
    Returns:
        Tuple of (token, token_id)
        
    Raises:
        ConnectionError: If unable to connect to proxy server
        TimeoutError: If request times out
        Exception: If token provisioning fails
    """
    # Get proxy URL from environment if not provided
    if proxy_url is None:
        is_valid, error_msg = validate_proxy_url()
        if not is_valid:
            raise Exception(error_msg)
        proxy_url = get_proxy_url()
    
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    print(f"🔗 Connecting to proxy: {proxy_url}")
    
    for attempt in range(RETRY_ATTEMPTS):
        try:
            print(f"🔄 Token provision attempt {attempt + 1}/{RETRY_ATTEMPTS}")
            
            response = requests.get(
                f"{proxy_url}/keys/provision", 
                headers=headers,
                timeout=REQUEST_TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                token = data["token"]
                token_id = data["token_id"]
                
                # For convenience, also set environment variable
                os.environ["HF_TOKEN"] = token
                
                print(f"✅ Token provisioned successfully: {token_id}")
                return token, token_id
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                print(f"❌ Provision failed: {error_msg}")
                
                if attempt == RETRY_ATTEMPTS - 1:  # Last attempt
                    raise Exception(f"Failed to provision token: {error_msg}")
                    
        except ConnectionError as e:
            error_msg = f"Connection failed to proxy server: {str(e)}"
            print(f"🔌 {error_msg}")
            
            if attempt == RETRY_ATTEMPTS - 1:  # Last attempt
                raise ConnectionError(f"Cannot connect to HF-Inferoxy at {proxy_url}. Please check if the server is running.")
            
        except Timeout as e:
            error_msg = f"Request timeout after {REQUEST_TIMEOUT}s: {str(e)}"
            print(f"⏰ {error_msg}")
            
            if attempt == RETRY_ATTEMPTS - 1:  # Last attempt
                raise TimeoutError(f"Timeout connecting to HF-Inferoxy. Server may be overloaded.")
                
        except RequestException as e:
            error_msg = f"Request error: {str(e)}"
            print(f"🚫 {error_msg}")
            
            if attempt == RETRY_ATTEMPTS - 1:  # Last attempt
                raise Exception(f"Network error connecting to proxy: {str(e)}")
        
        # Wait before retry
        if attempt < RETRY_ATTEMPTS - 1:
            print(f"⏱️ Retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)

def report_token_status(
    token_id: str, 
    status: str = "success", 
    error: Optional[str] = None,
    proxy_url: str = None,
    api_key: str = None
) -> bool:
    """
    Report token usage status back to the proxy server with timeout handling.
    
    Args:
        token_id: ID of the token to report (from get_proxy_token)
        status: Status to report ('success' or 'error')
        error: Error message if status is 'error'
        proxy_url: URL of the HF-Inferoxy server (optional, will use PROXY_URL env var if not provided)
        api_key: Your API key for authenticating with the proxy server
        
    Returns:
        True if report was accepted, False otherwise
    """
    # Get proxy URL from environment if not provided
    if proxy_url is None:
        is_valid, error_msg = validate_proxy_url()
        if not is_valid:
            print(f"❌ {error_msg}")
            return False
        proxy_url = get_proxy_url()
    
    payload = {"token_id": token_id, "status": status}
    
    if error:
        payload["error"] = error
        
        # Extract error classification based on actual HF error patterns
        error_type = None
        if "401 Client Error" in error:
            error_type = "invalid_credentials"
        elif "402 Client Error" in error and "exceeded your monthly included credits" in error:
            error_type = "credits_exceeded"
        elif "timeout" in error.lower() or "timed out" in error.lower():
            error_type = "timeout"
        elif "connection" in error.lower():
            error_type = "connection_error"
            
        if error_type:
            payload["error_type"] = error_type
    
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    print(f"📊 Reporting {status} for token {token_id}")
    
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = requests.post(
                f"{proxy_url}/keys/report", 
                json=payload, 
                headers=headers,
                timeout=REQUEST_TIMEOUT
            )
            
            if response.status_code == 200:
                print(f"✅ Status reported successfully")
                return True
            else:
                print(f"⚠️ Report failed: HTTP {response.status_code}")
                
        except ConnectionError as e:
            print(f"🔌 Report connection error: {str(e)}")
            
        except Timeout as e:
            print(f"⏰ Report timeout: {str(e)}")
            
        except RequestException as e:
            print(f"🚫 Report request error: {str(e)}")
            
        # Don't retry on last attempt
        if attempt < RETRY_ATTEMPTS - 1:
            print(f"⏱️ Retrying report in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)
    
    print(f"❌ Failed to report status after {RETRY_ATTEMPTS} attempts")
    return False
