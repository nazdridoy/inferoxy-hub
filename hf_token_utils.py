# hf_token_utils.py
import os
import requests
import json
from typing import Dict, Optional, Any, Tuple

def get_proxy_token(proxy_url: str = "http://scw.nazdev.tech:11155", api_key: str = None) -> Tuple[str, str]:
    """
    Get a valid token from the proxy server.
    
    Args:
        proxy_url: URL of the HF-Inferoxy server
        api_key: Your API key for authenticating with the proxy server
        
    Returns:
        Tuple of (token, token_id)
        
    Raises:
        Exception: If token provisioning fails
    """
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    response = requests.get(f"{proxy_url}/keys/provision", headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to provision token: {response.text}")
    
    data = response.json()
    token = data["token"]
    token_id = data["token_id"]
    
    # For convenience, also set environment variable
    os.environ["HF_TOKEN"] = token
    
    return token, token_id

def report_token_status(
    token_id: str, 
    status: str = "success", 
    error: Optional[str] = None,
    proxy_url: str = "http://scw.nazdev.tech:11155",
    api_key: str = None
) -> bool:
    """
    Report token usage status back to the proxy server.
    
    Args:
        token_id: ID of the token to report (from get_proxy_token)
        status: Status to report ('success' or 'error')
        error: Error message if status is 'error'
        proxy_url: URL of the HF-Inferoxy server
        api_key: Your API key for authenticating with the proxy server
        
    Returns:
        True if report was accepted, False otherwise
    """
    payload = {"token_id": token_id, "status": status}
    
    if error:
        payload["error"] = error
        
        # Extract error classification based on actual HF error patterns
        error_type = None
        if "401 Client Error" in error:
            error_type = "invalid_credentials"
        elif "402 Client Error" in error and "exceeded your monthly included credits" in error:
            error_type = "credits_exceeded"
            
        if error_type:
            payload["error_type"] = error_type
    
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        
    try:
        response = requests.post(f"{proxy_url}/keys/report", json=payload, headers=headers)
        return response.status_code == 200
    except Exception as e:
        # Silently fail to avoid breaking the client application
        # In production, consider logging this error
        return False
