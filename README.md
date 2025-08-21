---
title: HF-Inferoxy AI Hub
emoji: 🚀
colorFrom: purple
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# 🚀 HF-Inferoxy AI Hub

A comprehensive AI platform that combines conversational AI and text-to-image generation capabilities with intelligent HuggingFace API token management through HF-Inferoxy.

## ✨ Features

### 💬 Chat Assistant
- **🤖 Smart Conversations**: Advanced chat interface with streaming responses
- **🎯 Model Flexibility**: Support for any HuggingFace chat model
- **⚙️ Customizable Parameters**: Control temperature, top-p, max tokens, and system messages
- **🌐 Multi-Provider Support**: Works with Cerebras, Cohere, Groq, Together, and more

### 🎨 Image Generator
- **🖼️ Text-to-Image Generation**: Create stunning images from text descriptions
- **🎛️ Advanced Controls**: Fine-tune dimensions, inference steps, guidance scale, and seeds
- **🎯 Multiple Providers**: HF Inference, Fal.ai, Nebius, NScale, Replicate, Together
- **📱 Beautiful UI**: Modern interface with preset configurations and examples

### 🔄 Smart Token Management
- **🚀 Automatic Token Provisioning**: No manual token management required
- **⚡ Intelligent Rotation**: Automatic switching when tokens fail or reach limits
- **🛡️ Error Resilience**: Failed tokens are quarantined and replaced seamlessly
- **📊 Usage Tracking**: Comprehensive monitoring of token usage and errors

## 🛠️ Setup

### 1. HuggingFace Space Secrets

Add the following secrets to your HuggingFace Space:

- **Key**: `PROXY_KEY`
- **Value**: Your HF-Inferoxy proxy API key

- **Key**: `PROXY_URL`
- **Value**: Your HF-Inferoxy proxy server URL (e.g., `https://hf-proxy.example.com`)

### 2. HF-Inferoxy Server

The app will use the HF-Inferoxy server URL specified in the `PROXY_URL` secret.

### 3. Dependencies

The app requires:
- `gradio` - Modern web interface framework
- `huggingface-hub` - HuggingFace API integration
- `requests` - HTTP communication with the proxy
- `Pillow` - Image processing capabilities
- `torch` & `transformers` - Model support

## 🎯 How It Works

### Token Management Flow
1. **Token Provisioning**: The app requests a valid token from the HF-Inferoxy server
2. **API Calls**: Uses the provisioned token for HuggingFace API requests
3. **Status Reporting**: Reports token usage success/failure back to the proxy
4. **Automatic Rotation**: HF-Inferoxy handles token rotation and error management

### Chat Assistant
1. **Model Selection**: Choose any HuggingFace model with optional provider specification
2. **Conversation**: Engage in natural conversations with streaming responses
3. **Customization**: Adjust the AI's personality with system messages and parameters

### Image Generation
1. **Prompt Creation**: Write detailed descriptions of desired images
2. **Model & Provider**: Select from preset combinations or specify custom ones
3. **Parameter Tuning**: Fine-tune generation settings for optimal results
4. **Image Creation**: Generate high-quality images with automatic token management

## 🌟 Supported Models & Providers

### Chat Models

| Model | Provider | Description |
|-------|----------|-------------|
| `openai/gpt-oss-20b` | Fireworks AI, Cerebras, Groq | Fast general purpose model |
| `meta-llama/Llama-2-7b-chat-hf` | HF Inference | Chat-optimized model |
| `mistralai/Mistral-7B-Instruct-v0.2` | Featherless AI | Instruction following |
| `CohereLabs/c4ai-command-r-plus` | Cohere | Advanced language model |

### Image Models

| Model | Provider | Description |
|-------|----------|-------------|
| `Qwen/Qwen-Image` | Fal.ai, Replicate | Advanced image generation (default) |
| `black-forest-labs/FLUX.1-dev` | Nebius, Together | State-of-the-art image model |
| `stabilityai/stable-diffusion-xl-base-1.0` | HF Inference, NScale | High-quality SDXL model |

## 🎨 Usage Examples

### Chat Assistant

#### Basic Conversation
1. Go to the "💬 Chat Assistant" tab
2. Type your message in the chat input
3. Adjust parameters if needed (temperature, model, etc.)
4. Watch the AI respond with streaming text

#### Model Examples
```
# Auto provider (default - let HF choose best)
Model Name: openai/gpt-oss-20b

# Specific provider
Model Name: openai/gpt-oss-20b:fireworks-ai
System Message: You are a helpful coding assistant specializing in Python.
```

### Image Generation

#### Basic Image Creation
1. Go to the "🎨 Image Generator" tab
2. Enter your prompt: "A serene mountain lake at sunset, photorealistic, 8k"
3. Choose a model and provider
4. Click "🎨 Generate Image"

#### Advanced Settings
- **Dimensions**: 1024x1024 (must be divisible by 8)
- **Inference Steps**: 20-50 for good quality
- **Guidance Scale**: 7-10 for following prompts closely
- **Negative Prompt**: "blurry, low quality, distorted"

## ⚙️ Configuration Options

### Chat Parameters
- **System Message**: Define the AI's personality and behavior
- **Max New Tokens**: Control response length (1-4096)
- **Temperature**: Creativity level (0.1-2.0)
- **Top-p**: Response diversity (0.1-1.0)

### Image Parameters
- **Prompt**: Detailed description of desired image
- **Negative Prompt**: What to avoid in the image
- **Dimensions**: Width and height (256-2048, divisible by 8)
- **Inference Steps**: Quality vs speed trade-off (10-100)
- **Guidance Scale**: Prompt adherence (1.0-20.0)
- **Seed**: Reproducibility (-1 for random)

## 🎯 Provider-Specific Features

### Chat Providers
- **Auto**: Let HuggingFace choose the best provider (default)
- **Fireworks AI**: Fast and reliable inference service
- **Cerebras**: High-performance inference with low latency
- **Cohere**: Advanced language models with multilingual support
- **Groq**: Ultra-fast inference, optimized for speed
- **Together**: Collaborative AI hosting, wide model support
- **Featherless AI**: Specialized fine-tuned models

### Image Providers
- **HF Inference**: Core API with comprehensive model support
- **Fal.ai**: High-quality image generation with fast processing
- **Nebius**: Cloud-native services with enterprise features
- **NScale**: Optimized inference performance
- **Replicate**: Collaborative AI hosting with version control
- **Together**: Fast inference service with wide model support

## 💡 Tips for Better Results

### Chat Tips
- **Clear Instructions**: Be specific about what you want
- **System Messages**: Set context and personality upfront
- **Model Selection**: Choose appropriate models for your task
- **Parameter Tuning**: Lower temperature for factual responses, higher for creativity

### Image Tips
- **Detailed Prompts**: Use specific, descriptive language
- **Style Keywords**: Include art style, lighting, and quality descriptors
- **Negative Prompts**: Specify what you don't want to avoid common issues
- **Aspect Ratios**: Consider the subject when choosing dimensions
- **Provider Testing**: Try different providers for varied artistic styles

### Example Prompts

#### Chat Examples
```
# Using auto provider (default)
Model: openai/gpt-oss-20b
Prompt: "Explain quantum computing in simple terms"

# Using specific provider
Model: openai/gpt-oss-20b:fireworks-ai  
Prompt: "Help me debug this Python code: [paste code]"

# Other example prompts:
"Write a creative story about a time-traveling cat"
"What are the pros and cons of renewable energy?"
```

#### Image Examples
```
"A majestic dragon flying over a medieval castle, epic fantasy art, detailed, 8k"
"A serene Japanese garden with cherry blossoms, zen atmosphere, peaceful, high quality"
"A futuristic cityscape with flying cars and neon lights, cyberpunk style, cinematic"
"Portrait of a wise old wizard with flowing robes, magical aura, fantasy character art"
```

## 🔒 Security & Authentication

### RBAC System
- All operations require authentication with the HF-Inferoxy proxy server
- API keys are managed securely through HuggingFace Space secrets
- No sensitive information is logged or exposed

### Token Security
- Tokens are automatically rotated when they fail or reach limits
- Failed tokens are quarantined to prevent repeated failures
- Usage is tracked comprehensively for monitoring and optimization

## 🐛 Troubleshooting

### Common Issues

#### Setup Issues
1. **PROXY_KEY Missing**: Ensure the secret is set in your HuggingFace Space settings
2. **PROXY_URL Missing**: Ensure the proxy server URL secret is set in your HuggingFace Space settings
3. **Connection Errors**: Verify the HF-Inferoxy server is accessible
4. **Import Errors**: Check that all dependencies are properly installed

#### Chat Issues
1. **No Response**: Check model name format and provider availability
2. **Slow Responses**: Try different providers or smaller models
3. **Poor Quality**: Adjust temperature and top-p parameters

#### Image Issues
1. **Generation Fails**: Verify model supports text-to-image generation
2. **Dimension Errors**: Ensure width and height are divisible by 8
3. **Poor Quality**: Increase inference steps or adjust guidance scale

### Error Types
- **401 Errors**: Authentication issues (handled automatically by token rotation)
- **402 Errors**: Credit limit exceeded (reported to proxy for token management)
- **Network Errors**: Connection issues (reported to proxy for monitoring)
- **Model Errors**: Invalid model or provider combinations

## 📚 Additional Resources

- **[HF-Inferoxy Documentation](https://nazdridoy.github.io/hf-inferoxy/)**: Complete platform documentation
- **[HuggingFace Hub Integration Guide](https://nazdridoy.github.io/hf-inferoxy/huggingface-hub-integration/)**: Detailed integration instructions
- **[Provider Examples](https://nazdridoy.github.io/hf-inferoxy/examples/)**: Code examples for different providers
- **[Gradio Documentation](https://gradio.app/docs/)**: Interface framework documentation

## 🤝 Contributing

This application is part of the HF-Inferoxy ecosystem. For contributions or issues:

1. Review the [HF-Inferoxy documentation](https://nazdridoy.github.io/hf-inferoxy/)
2. Test with different models and providers
3. Report any issues or suggest improvements
4. Contribute examples and use cases

## 🚀 Advanced Usage

### Environment Variables

You can customize the proxy URL using environment variables:

```python
import os
os.environ["HF_PROXY_URL"] = "http://your-proxy-server:8000"
```

### Custom Providers

The app supports any provider that works with HF-Inferoxy. Simply specify the provider name when entering model information.

### Batch Operations

For multiple operations, consider the token reuse patterns documented in the HF-Inferoxy integration guide.

## 📄 License

This project is part of the HF-Inferoxy ecosystem. Please refer to the main project for licensing information.

---

**Built with ❤️ using [HF-Inferoxy](https://nazdridoy.github.io/hf-inferoxy/) for intelligent token management**

**Ready to explore AI? Start chatting or generating images above! 🚀**
