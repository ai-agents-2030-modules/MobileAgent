import base64
import requests
import os

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def inference_chat(chat, API_TOKEN, model, token_storage = None):    
    api_url = 'https://api.openai.com/v1/chat/completions'
    if os.getenv("OPENAI_BASE_URL"):
        api_url = os.getenv("OPENAI_BASE_URL") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_TOKEN}"
    }
    data = {
        "model": model,
        "messages": [],
        "max_tokens": 2048,
    }

    for role, content in chat:
        data["messages"].append({"role": role, "content": content})

    counter = 0
    while counter < 3:
        try:
            res = requests.post(api_url, headers=headers, json=data)
            usage = res.json()["usage"]
            res = res.json()['choices'][0]['message']['content']
            if token_storage:
                token_storage["prompt_tokens"] += usage["prompt_tokens"]
                token_storage["completion_tokens"] += usage["completion_tokens"]
        except Exception as e:
            print("Network Error:")
            print(e)
            try:
                print(res)
            except:
                pass
        else:
            break
        counter += 1
    
    if counter == 3:
        return "ERROR"

    return res
