import subprocess
# Installing flash_attn
subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)

import gradio as gr
from PIL import Image 
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 
from transformers import TextIteratorStreamer
import time
from threading import Thread
import torch
import spaces

model_id = "microsoft/Phi-3-vision-128k-instruct" 
##model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto")
##processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True) 
model.to("cuda:0")



PLACEHOLDER = """
<div style="padding: 30px; text-align: center; display: flex; flex-direction: column; align-items: center;">   
   <h1 style="font-size: 28px; margin-bottom: 2px; opacity: 0.55;"></h1>
   <p style="font-size: 18px; margin-bottom: 2px; opacity: 0.65;"></p>
</div>
"""




@spaces.GPU
def open_image4xl(message, history):
    print(f'message is - {message}')
    print(f'history is - {history}')
    if message["files"]:
        # message["files"][-1] is a Dict or just a string
        if type(message["files"][-1]) == dict:
            image = message["files"][-1]["path"]
        else:
            image = message["files"][-1]
    else:
        
        for hist in history:
            if type(hist[0]) == tuple:
                image = hist[0][0]
    try:
        if image is None:
            # Handle the case where image is None
            raise gr.Error("You need to upload an image for model to work. Close the error and try again with an Image.")
    except NameError:
        # Handle the case where 'image' is not defined at all
        raise gr.Error("You need to upload an image for model to work. Close the error and try again with an Image.")

    conversation = []
    flag=False
    for user, assistant in history:
        if assistant is None:
            #pass
            flag=True
            conversation.extend([{"role": "user", "content":""}])
            continue
        if flag==True:
            conversation[0]['content'] = f"<|image_1|>\n{user}"   
            conversation.extend([{"role": "assistant", "content": assistant}])
            flag=False
            continue
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])

    if len(history) == 0:
        conversation.append({"role": "user", "content": f"<|image_1|>\n{message['text']}"})
    else:
        conversation.append({"role": "user", "content": message['text']})
    print(f"prompt is -\n{conversation}")
    prompt = processor.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    image = Image.open(image)
    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0") 

    streamer = TextIteratorStreamer(processor, **{"skip_special_tokens": True, "skip_prompt": True, 'clean_up_tokenization_spaces':False,}) 
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=1024, do_sample=False, temperature=0.0, eos_token_id=processor.tokenizer.eos_token_id,)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    buffer = ""
    for new_text in streamer:
        buffer += new_text
        yield buffer


chatbot=gr.Chatbot(scale=1, placeholder=PLACEHOLDER)
chat_input = gr.MultimodalTextbox(interactive=True, file_types=["image"], placeholder="Upload the image or file...", show_label=False)
with gr.Blocks(fill_height=True, theme="xiaobaiyuan/theme_brief" ) as demo:
    gr.ChatInterface(
    fn=open_image4xl,
    title="",
    examples=[{"text": "Derive the full equation.", "files": ["./assets/images/1.png"]},
              {"text": "Explain the stage of the plant.", "files": ["./assets/images/5.png"]},
              {"text": "Give the sum of the digits.", "files": ["./assets/images/2.png"]},
              {"text": "What is the symbol named, give it's value ?.", "files": ["./assets/images/4.png"]},
              {"text": "What are the colors involved in the image ?", "files": ["./assets/images/3.png"]},
             ],
    description="",
    stop_btn="Cancel Generation",
    multimodal=True,
    textbox=chat_input,
    chatbot=chatbot,
    cache_examples=False,
    examples_per_page=3
    )

demo.queue()
demo.launch(debug=True, quiet=True)
