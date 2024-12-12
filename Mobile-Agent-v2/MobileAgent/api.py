import base64
import requests

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def inference_chat(chat, model, api_url, token, token_storage = None):    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    data = {
        "model": model,
        "messages": [],
        "max_tokens": 2048,
        'temperature': 0.0,
        "seed": 1234
    }

    for role, content in chat:
        data["messages"].append({"role": role, "content": content})

    counter = 0
    while counter < 3:
        counter += 1
        try:
            res = requests.post(api_url, headers=headers, json=data)
            res_json = res.json()
            res_content = res_json['choices'][0]['message']['content']
            if token_storage:
                token_storage["prompt_tokens"] += res_json["usage"]["prompt_tokens"]
                token_storage["completion_tokens"] += res_json["usage"]["completion_tokens"]
        except Exception as e:
            print("Network Error:")
            print(e)
            try:
                print(res)
            except:
                pass
        else:
            break
    
    if counter == 3:
        return "ERROR"

    return res_content
