# ShadowJourney API Client

## Overview

The ShadowJourney API Client provides a comprehensive interface for interacting with ShadowJourney API. This library supports both synchronous and asynchronous methods to accommodate different programming styles and use cases.

## Installation

You can install the package directly from PyPI using the following command:  
```bash
pip install shadowjourney
```

Alternatively, you can install the latest (possibly unstable) version from GitHub:  
```bash
pip install git+https://github.com/shadowjourneyy/shadowjourney.git
```

## Chat Completions

### Synchronous Usage

```python
from shadowjourney import API

client = API(api_key="your_api_key")

# Non-Streaming
response = client.chat.completions.create(
    model="gpt-40",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing."}
    ],
    stream=False
)

print(response.choices[0].message.content)

# Streaming Response
chunks = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing."}
    ],
    stream=True,
    model="gpt-4o"
)
for chunk in chunks:
    print(chunk.choices[0].delta.content, end="", flush=True)
```

### Asynchronous Usage

```python
import asyncio
from shadowjourney import AsyncAPI

client = AsyncAPI(api_key="your_api_key")

# Non Streaming
async def non_streaming():
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain quantum computing."}
        ],
    )
    print(response.choices[0].message.content)

async def streaming():
    chunks = await client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True,
        model=CHATMODEL
    )
    async for chunk in chunks:
        print(chunk.choices[0].delta.content)

```

## Image Generation

### Synchronous Usage

```python
from shadowjourney import API

client = API(api_key="your_api_key")

# Generate an image
response = client.images.generate(
    prompt="A futuristic cityscape at sunset",
    model="dall-e-3",
    size="1024x1024",
    n=1
)

print(response.data[0].url)
```

### Asynchronous Usage

```python
import asyncio
from shadowjourney import AsyncAPI

client = AsyncAPI(api_key="your_api_key")

async def main():
    # Generate an image
    response = await client.images.generate(
        prompt="A futuristic cityscape at sunset",
        model="dall-e-3",
        size="1024x1024",
        n=1
    )
    print(response.data[0].url)
```
