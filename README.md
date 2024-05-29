---
title: ImageGPT 4XL
emoji: ☄️
colorFrom: pink
colorTo: gray
sdk: gradio
sdk_version: 4.31.5
app_file: app.py
pinned: true
license: creativeml-openrail-m
short_description: Analyze, solve, ask ..
---

🚀Check out the configuration reference at : https://huggingface.co/docs/hub/spaces-config-reference

🚀Huggingface space : https://huggingface.co/spaces/prithivMLmods/ImageGPT-4XL

🚀The GitHub Model Workspace : 

    1. Install the client if you don't already have it installed.
    
    $ pip install gradio_client
    
    2. Find the API endpoint below corresponding to your desired function in the app. Copy the code snippet, replacing the placeholder values with your own input data. If this is a private Space, you may need to pass your Hugging Face token as well (read more). Or
    
    to automatically generate your API requests.
    api_name: /chat
    
    from gradio_client import Client
    
    client = Client("prithivMLmods/ImageGPT-4XL")
    result = client.predict(
    		message={"text":"","files":[]},
    		api_name="/chat"
    )
    print(result)
    
    Accepts 1 parameter:
    
    message Dict(text: str, files: List[filepath]) Default: {"text":"","files":[]}
    
    The input value that is provided in the "parameter_1" Multimodaltextbox component.
    Returns 1 element
    
    Dict(text: str, files: List[filepath])
    
    The output value that appears in the "value_1" Multimodaltextbox component.

## 🌚The image given to the gpt to process

![alt text](assets/11.png)

## 🌝The processed from the model

![alt text](assets/12.png)

## The examples given to the model: 

| ![Image 1](assets/images/1.png) | ![Image 2](assets/images/2.png) |
|---------------------------------|---------------------------------|
| ![Image 3](assets/images/3.png) | ![Image 4](assets/images/4.png) |

## Requirements.txt [ PyPI ]

    #flash_attn
    accelerate
    git+https://github.com/huggingface/transformers.git@main
    spaces
    torchvision
    Pillow


.

.

.
