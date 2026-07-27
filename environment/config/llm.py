from openai import OpenAI

from environment.config.config import config


def get_client(model_prefix=None):
    """Get OpenAI client with appropriate credentials based on model prefix"""
    # Get model-specific API key and base URL if available, otherwise use defaults
    if model_prefix and f"{model_prefix}_api_key" in config['llm']:
        api_key = config['llm'][f"{model_prefix}_api_key"]
        base_url = config['llm'][f"{model_prefix}_base_url"]
    else:
        api_key = config['llm']['api_key']
        base_url = config['llm']['base_url']
    
    # Create and return client
    return OpenAI(api_key=api_key, base_url=base_url)

def deepseek(model="deepseek-v3", system=None, user=None, messages=None):
    # Get client for deepseek
    client = get_client("deepseek")
    
    if messages is not None:
        pass
    else:
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        if user is not None:
            messages.append({"role": "user", "content": user})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1,
        response_format={"type": "json_object"}
    )
    return response

def claude(model="claude-3-7-sonnet-20250219", system=None, user=None, messages=None):
    # Get client for claude
    client = get_client("claude")
    
    if messages is not None:
        pass
    else:
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        if user is not None:
            messages.append({"role": "user", "content": user})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1
    )
    return response

def gemini(model="gemini-2.5-flash", system=None, user=None, messages=None):
    # Get client for gemini
    client = get_client("gemini")
    
    if messages is not None:
        pass
    else:
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        if user is not None:
            messages.append({"role": "user", "content": user})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1
    )
    return response

def twelvelabs_client():
    """Build a TwelveLabs client from the ``twelvelabs`` credentials in config.yml.

    The ``twelvelabs`` SDK is imported lazily so it is only required when the
    Pegasus backend is actually used.
    """
    from twelvelabs import TwelveLabs

    api_key = config['llm'].get('twelvelabs_api_key')
    if not api_key:
        raise ValueError(
            "TwelveLabs API key not configured. Set llm.twelvelabs_api_key in "
            "environment/config/config.yml (free key at https://twelvelabs.io)."
        )

    base_url = config['llm'].get('twelvelabs_base_url') or None
    if base_url:
        return TwelveLabs(api_key=api_key, base_url=base_url)
    return TwelveLabs(api_key=api_key)

def pegasus(video, prompt, model="pegasus1.5", max_tokens=2048, temperature=None):
    """Analyze a video with TwelveLabs Pegasus and return the generated text.

    ``video`` is either a public URL string or a dict describing a
    TwelveLabs video context, e.g. ``{"type": "url", "url": "..."}`` or
    ``{"type": "asset_id", "asset_id": "..."}``. Unlike the chat helpers
    above, Pegasus understands the video natively (frames + audio), so no
    transcription step is required.
    """
    if isinstance(video, str):
        video = {"type": "url", "url": video}

    return twelvelabs_client().analyze(
        model_name=model,
        video=video,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )

def gpt(model="gpt-4o", system=None, user=None, messages=None):
    # Get client for gpt
    client = get_client("gpt")
    
    if messages is not None:
        pass
    else:
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        if user is not None:
            messages.append({"role": "user", "content": user})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1
    )
    return response