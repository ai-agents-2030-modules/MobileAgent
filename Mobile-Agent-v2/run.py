import os
import time
import copy
import torch
from PIL import Image, ImageDraw
import base64
import requests

from MobileAgent.api import inference_chat
from MobileAgent.text_localization import ocr
from MobileAgent.icon_localization import det
from MobileAgent.controller import get_screenshot, tap, slide, type, back, home
from MobileAgent.prompt import get_action_prompt, get_reflect_prompt, get_memory_prompt, get_process_prompt
from MobileAgent.chat import init_action_chat, init_reflect_chat, init_memory_chat, add_response, add_response_two_image

from load_models import load_cached_model
from modelscope.utils.constant import Tasks
from modelscope import snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from dashscope import MultiModalConversation
import dashscope
import concurrent

import json
import queue
import shutil
import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8")

parser = argparse.ArgumentParser()
parser.add_argument("--instruction", type=str)
parser.add_argument("--adb_path", type=str)
parser.add_argument("--openai_api_model", default="gpt-4o")
parser.add_argument("--openai_api_key", type=str)
parser.add_argument("--qwen_api_key", type=str)
# Add necessary arguments for benchmark
parser.add_argument("--lang", default="ENG")
parser.add_argument("--output_dir")
parser.add_argument("--max_rounds", type=int, default=20)
parser.add_argument("--device", type=str)
args = parser.parse_args()


def print_and_log_error(error_message):
    print(error_message)
    error_log = [{"error_message": error_message}]
    filename = args.output_dir + "/error.json"
    # Check if the file already exists
    if not os.path.exists(filename):
        # If the file does not exist, create it and write the JSON data
        with open(filename, "w", encoding="utf-8") as logfile:
            json.dump(error_log, logfile, ensure_ascii=False)


####################################### Edit your Setting #########################################
start_time_initial = time.time()

# Your ADB path
adb_command_prefix = args.adb_path + " -s " + args.device

# Your instruction
instruction = args.instruction

# Your GPT-4o API URL
API_url = "https://api.openai.com/v1/chat/completions"
if os.getenv("OPENAI_BASE_URL"):
    API_url = os.getenv("OPENAI_BASE_URL") + "/chat/completions"

# Your GPT-4o API Token
token = args.openai_api_key

# Choose between "api" and "local". api: use the qwen api. local: use the local qwen checkpoint
caption_call_method = "api" if args.qwen_api_key != "self_hosted" else "local"

# Choose between "qwen-vl-plus" and "qwen-vl-max" if use api method. Choose between "qwen-vl-chat" and "qwen-vl-chat-int4" if use local method.
caption_model = "qwen-vl-plus" if caption_call_method == "api" else "qwen-vl-chat"

# If you choose the api caption call method, input your Qwen api here
qwen_api = args.qwen_api_key

# You can add operational knowledge to help Agent operate more accurately.
add_info = 'If you want to tap an icon of an app, use the action "Open app". If you want to exit an app, use the action "Home"'

# Reflection Setting: If you want to improve the operating speed, you can disable the reflection agent. This may reduce the success rate.
reflection_switch = True

# Memory Setting: If you want to improve the operating speed, you can disable the memory unit. This may reduce the success rate.
memory_switch = True
###################################################################################################


def get_all_files_in_folder(folder_path):
    file_list = []
    for file_name in os.listdir(folder_path):
        file_list.append(file_name)
    return file_list


def draw_coordinates_on_image(image_path, coordinates):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    point_size = 10
    for coord in coordinates:
        draw.ellipse((coord[0] - point_size, coord[1] - point_size, coord[0] + point_size, coord[1] + point_size), fill='red')
    output_image_path = screenshot + "/output_image.png"
    image.save(output_image_path)
    return output_image_path


def crop(image, box, i):
    image = Image.open(image)
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    if x1 >= x2-10 or y1 >= y2-10:
        return
    cropped_image = image.crop((x1, y1, x2, y2))
    cropped_image.save(f"./temp_{args.device}/{i}.jpg")


def inference_chat_for_qwen(chat, model, api_url):
    headers = {"Content-Type": "application/json"}

    data = {
        "model": model,
        "messages": [],
        "max_tokens": 1024,
        "temperature": 0.0,
        "seed": 1234,
    }

    for role, content in chat:
        data["messages"].append({"role": role, "content": content})

    counter = 0
    while counter < 3:
        counter += 1
        try:
            res = requests.post(api_url, headers=headers, json=data)
            res_json = res.json()
            res_content = res_json["choices"][0]["message"]["content"]
        except:
            print("Network Error:")
            try:
                print(res.json())
            except:
                print("Request Failed")
        else:
            break

    if counter == 3:
        return "ERROR"

    return res_content


def generate_local(tokenizer, model, image_file, query):
    chat_action = add_response("user", query, [], image_file)
    response = inference_chat_for_qwen(
        chat_action,
        "qwen-vl-chat",
        os.getenv("QWEN_API_URL", "http://127.0.0.1:7001/v1/chat/completions"),
    )
    return response


def process_image(image, query):
    dashscope.api_key = qwen_api
    image = "file://" + image
    messages = [{
        'role': 'user',
        'content': [
            {
                'image': image
            },
            {
                'text': query
            },
        ]
    }]
    response = MultiModalConversation.call(model=caption_model, messages=messages)

    try:
        usage = (
            8e-06
            * (
                response["usage"]["input_tokens"]
                + response["usage"]["output_tokens"]
                + response["usage"]["image_tokens"]
            )
            * 0.14
        )  # 0.14 is the currency convert rate from RMB to dollars
        response = response["output"]["choices"][0]["message"]["content"][0]["text"]
    except Exception as e:
        print("Error in process_image")
        print(e)
        print(response)
        response = "ERROR"
        usage = 0

    return response, usage


icon_error_counter = queue.Queue()


def process_image_mini(image, query):
    # OpenAI API Key
    api_key = token

    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # Getting the base64 string
    base64_image = encode_image(image)

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low",
                        },
                    },
                ],
            }
        ],
    }
    counter = 0
    while True:
        counter += 1
        try:
            response = requests.post(API_url, headers=headers, json=payload)
            if response.status_code != 200:
                raise Exception("request error")
            response = response.json()
            usage = (
                1.5e-07 * response["usage"]["prompt_tokens"]
                + 6e-07 * response["usage"]["completion_tokens"]
            )
            response = response["choices"][0]["message"]["content"]
            return response, usage
        except Exception as e:
            if counter < 3:
                continue  # try again
            else:
                if icon_error_counter.qsize() < 5:
                    icon_error_counter.put(1)
                    return "This is an icon.", 0
                else:
                    print(response)
                    print("Error in process_image")
                    print(e)
                    return "ERROR", 0


def generate_api(images, query):
    icon_map = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_image_mini, image, query): i
            for i, image in enumerate(images)
        }

        api_usage = 0
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            response, usage = future.result()
            if response == "ERROR":
                return "ERROR"
            icon_map[i + 1] = response
            api_usage += usage

    return icon_map, api_usage


def merge_text_blocks(text_list, coordinates_list):
    merged_text_blocks = []
    merged_coordinates = []

    sorted_indices = sorted(range(len(coordinates_list)), key=lambda k: (coordinates_list[k][1], coordinates_list[k][0]))
    sorted_text_list = [text_list[i] for i in sorted_indices]
    sorted_coordinates_list = [coordinates_list[i] for i in sorted_indices]

    num_blocks = len(sorted_text_list)
    merge = [False] * num_blocks

    for i in range(num_blocks):
        if merge[i]:
            continue

        anchor = i

        group_text = [sorted_text_list[anchor]]
        group_coordinates = [sorted_coordinates_list[anchor]]

        for j in range(i+1, num_blocks):
            if merge[j]:
                continue

            if abs(sorted_coordinates_list[anchor][0] - sorted_coordinates_list[j][0]) < 10 and \
            sorted_coordinates_list[j][1] - sorted_coordinates_list[anchor][3] >= -10 and sorted_coordinates_list[j][1] - sorted_coordinates_list[anchor][3] < 30 and \
            abs(sorted_coordinates_list[anchor][3] - sorted_coordinates_list[anchor][1] - (sorted_coordinates_list[j][3] - sorted_coordinates_list[j][1])) < 10:
                group_text.append(sorted_text_list[j])
                group_coordinates.append(sorted_coordinates_list[j])
                merge[anchor] = True
                anchor = j
                merge[anchor] = True

        merged_text = "\n".join(group_text)
        min_x1 = min(group_coordinates, key=lambda x: x[0])[0]
        min_y1 = min(group_coordinates, key=lambda x: x[1])[1]
        max_x2 = max(group_coordinates, key=lambda x: x[2])[2]
        max_y2 = max(group_coordinates, key=lambda x: x[3])[3]

        merged_text_blocks.append(merged_text)
        merged_coordinates.append([min_x1, min_y1, max_x2, max_y2])

    return merged_text_blocks, merged_coordinates


def get_perception_infos(adb_command_prefix, screenshot_file):
    for retry in range(2):
        try:
            get_screenshot(adb_command_prefix, screenshot)
            width, height = Image.open(screenshot_file).size
            text, coordinates = ocr(screenshot_file, ocr_detection, ocr_recognition)
            break
        except Exception as err:
            if retry == 0 and str(err) == "modelscope error: No text detected":
                # To wait 10 seconds if the UI is not ready
                print("modelscope error: No text detected: Will retry in 10 seconds")
                time.sleep(10)
                continue
            else:
                raise err
    text, coordinates = merge_text_blocks(text, coordinates)

    center_list = [[(coordinate[0]+coordinate[2])/2, (coordinate[1]+coordinate[3])/2] for coordinate in coordinates]
    draw_coordinates_on_image(screenshot_file, center_list)

    perception_infos = []
    for i in range(len(coordinates)):
        perception_info = {"text": "text: " + text[i], "coordinates": coordinates[i]}
        perception_infos.append(perception_info)

    coordinates = det(screenshot_file, "icon", groundingdino_model)

    for i in range(len(coordinates)):
        perception_info = {"text": "icon", "coordinates": coordinates[i]}
        perception_infos.append(perception_info)

    image_box = []
    image_id = []
    for i in range(len(perception_infos)):
        if perception_infos[i]['text'] == 'icon':
            image_box.append(perception_infos[i]['coordinates'])
            image_id.append(i)

    for i in range(len(image_box)):
        crop(screenshot_file, image_box[i], image_id[i])

    images = get_all_files_in_folder(temp_file)
    if len(images) > 0:
        images = sorted(images, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        image_id = [int(image.split('/')[-1].split('.')[0]) for image in images]
        icon_map = {}
        prompt = 'This image is an icon from a phone screen. Please briefly describe the shape and color of this icon in one sentence.'
        if caption_call_method == "local":
            for i in range(len(images)):
                image_path = os.path.join(temp_file, images[i])
                icon_width, icon_height = Image.open(image_path).size
                if icon_height > 0.8 * height or icon_width * icon_height > 0.2 * width * height:
                    des = "None"
                else:
                    des = generate_local(tokenizer, model, image_path, prompt)
                icon_map[i+1] = des
        else:
            for i in range(len(images)):
                images[i] = os.path.join(temp_file, images[i])
            icon_map, api_usage = generate_api(images, prompt)
            if icon_map == "ERROR":
                return "ERROR", width, height
        for i, j in zip(image_id, range(1, len(image_id) + 1)):
            if icon_map.get(j):
                perception_infos[i]['text'] = "icon: " + icon_map[j]

    for i in range(len(perception_infos)):
        perception_infos[i]['coordinates'] = [int((perception_infos[i]['coordinates'][0]+perception_infos[i]['coordinates'][2])/2), int((perception_infos[i]['coordinates'][1]+perception_infos[i]['coordinates'][3])/2)]

    return perception_infos, width, height, api_usage

### Load caption model ###
device = "cuda"
torch.manual_seed(1234)
if caption_call_method == "local":
    if caption_model == "qwen-vl-chat":
        # qwen_dir = snapshot_download('qwen/Qwen-VL-Chat', revision='v1.1.0')
        # model = AutoModelForCausalLM.from_pretrained(qwen_dir, device_map=device, trust_remote_code=True).eval()
        # model.generation_config = GenerationConfig.from_pretrained(qwen_dir, trust_remote_code=True)
        model = None
    elif caption_model == "qwen-vl-chat-int4":
        qwen_dir = snapshot_download("qwen/Qwen-VL-Chat-Int4", revision="v1.0.0")
        model = AutoModelForCausalLM.from_pretrained(
            qwen_dir, device_map=device, trust_remote_code=True, use_safetensors=True
        ).eval()
        model.generation_config = GenerationConfig.from_pretrained(
            qwen_dir, trust_remote_code=True, do_sample=False
        )
    else:
        print_and_log_error(
            'If you choose local caption method, you must choose the caption model from "Qwen-vl-chat" and "Qwen-vl-chat-int4"'
        )
        sys.exit(3)
    tokenizer = None
    # tokenizer = AutoTokenizer.from_pretrained(qwen_dir, trust_remote_code=True)
elif caption_call_method == "api":
    pass
else:
    print("You must choose the caption model call function from \"local\" and \"api\"")
    print(3)
    sys.exit(3)


### Load ocr and icon detection model ###
groundingdino_model = load_cached_model(
    cache_path=r"~\.cache\modelscope\hub\AI-ModelScope\GroundingDINO",
    model_name="AI-ModelScope/GroundingDINO",
    pipeline_name="grounding-dino-task",
    snapshot_kwargs={"revision": "v1.0.0"}
)

ocr_detection = load_cached_model(
    cache_path=r"~\.cache\modelscope\hub\damo/cv_resnet18_ocr-detection-line-level_damo",
    model_name="damo/cv_resnet18_ocr-detection-line-level_damo",
    pipeline_name=Tasks.ocr_detection
)

ocr_recognition = load_cached_model(
    cache_path=r"~\.cache\modelscope\hub\damo/cv_convnextTiny_ocr-recognition-document_damo",
    model_name="damo/cv_convnextTiny_ocr-recognition-document_damo",
    pipeline_name=Tasks.ocr_recognition
)

thought_history = []
summary_history = []
action_history = []
summary = ""
action = ""
completed_requirements = ""
memory = ""
insight = ""
temp_file = "temp_" + args.device
screenshot = "screenshot_" + args.device
if not os.path.exists(temp_file):
    os.mkdir(temp_file)
else:
    shutil.rmtree(temp_file)
    os.mkdir(temp_file)
if not os.path.exists(screenshot):
    os.mkdir(screenshot)
error_flag = False

end_time_initial = time.time()
elapsed_time_initial = end_time_initial - start_time_initial
task_complete = False
total_prompt_tokens, total_completion_tokens = 0, 0
error_code = 0
width, height = 0, 0
iter = 0
api_usage = 0
perception_api_usage = 0
benchmark_log = []
start_time_exec = time.time()
try:
    while iter < args.max_rounds:
        iter += 1
        if iter == 1:
            screenshot_file = screenshot + "/screenshot.jpg"
            perception_infos, width, height, api_usage = get_perception_infos(
                adb_command_prefix, screenshot_file
            )
            if perception_infos == "ERROR":
                error_code = 3
                print_and_log_error("ERROR: get perception infos failed")
                break
            shutil.rmtree(temp_file)
            os.mkdir(temp_file)

            keyboard = False
            keyboard_height_limit = 0.9 * height
            for perception_info in perception_infos:
                if perception_info["coordinates"][1] < keyboard_height_limit:
                    continue
                if "ADB Keyboard" in perception_info["text"]:
                    keyboard = True
                    break
        perception_api_usage += api_usage

        token_storage = {"prompt_tokens": 0, "completion_tokens": 0}
        reflect_back = False

        prompt_action = get_action_prompt(
            instruction,
            perception_infos,
            width,
            height,
            keyboard,
            summary_history,
            action_history,
            summary,
            action,
            add_info,
            error_flag,
            completed_requirements,
            memory,
        )
        chat_action = init_action_chat()
        chat_action = add_response("user", prompt_action, chat_action, screenshot_file)

        output_action = inference_chat(
            chat_action, args.openai_api_model, API_url, token, token_storage
        )
        if output_action == "ERROR":
            print_and_log_error("ERROR: Inference output_action failed.")
            error_code = 3
            break
        shutil.copy(
            src=screenshot_file, dst=os.path.join(args.output_dir, f"{iter-1}.png")
        )
        thought = (
            output_action.split("### Thought ###")[-1]
            .split("### Action ###")[0]
            .replace("\n", " ")
            .replace(":", "")
            .replace("  ", " ")
            .strip()
        )
        summary = (
            output_action.split("### Operation ###")[-1]
            .replace("\n", " ")
            .replace("  ", " ")
            .strip()
        )
        action = (
            output_action.split("### Action ###")[-1]
            .split("### Operation ###")[0]
            .replace("\n", " ")
            .replace("  ", " ")
            .strip()
        )
        chat_action = add_response("assistant", output_action, chat_action)
        status = "#" * 50 + " Decision " + "#" * 50
        print(status)
        print(output_action)
        print("#" * len(status))

        if memory_switch:
            prompt_memory = get_memory_prompt(insight)
            chat_action = add_response("user", prompt_memory, chat_action)
            output_memory = inference_chat(
                chat_action, args.openai_api_model, API_url, token, token_storage
            )
            if output_memory == "ERROR":
                print_and_log_error("ERROR: Inference output_memory failed.")
                error_code = 2
            chat_action = add_response("assistant", output_memory, chat_action)
            status = "#" * 50 + " Memory " + "#" * 50
            print(status)
            print(output_memory)
            print("#" * len(status))
            output_memory = (
                output_memory.split("### Important content ###")[-1]
                .split("\n\n")[0]
                .strip()
                + "\n"
            )
            if "None" not in output_memory and output_memory not in memory:
                memory += output_memory

        action_log = [
            "",
            {"detail_type": "string", "detail": ""},
        ]  # second element for action details based act_name
        if "Open app" in action:
            app_name = action.split("(")[-1].split(")")[0]
            text, coordinate = ocr(screenshot_file, ocr_detection, ocr_recognition)
            tap_coordinate = [0, 0]
            for ti in range(len(text)):
                if app_name == text[ti]:
                    name_coordinate = [
                        int((coordinate[ti][0] + coordinate[ti][2]) / 2),
                        int((coordinate[ti][1] + coordinate[ti][3]) / 2),
                    ]
                    action_log[0] = f'Open app "({app_name})"'
                    action_log[1]["detail_type"] = "coordinates"
                    action_log[1]["detail"] = [
                        name_coordinate[0],
                        name_coordinate[1] - int(coordinate[ti][3] - coordinate[ti][1]),
                    ]
                    tap(
                        adb_command_prefix,
                        name_coordinate[0],
                        name_coordinate[1] - int(coordinate[ti][3] - coordinate[ti][1]),
                    )  #

        elif "Tap" in action:
            try:
                coordinate = action.split("(")[-1].split(")")[0].split(", ")
                x, y = int(coordinate[0]), int(coordinate[1])
            except:
                print_and_log_error("ERROR: Action parameter cannot be parsed.")
                error_code = 2
                break
            action_log[0] = "Tap"
            action_log[1]["detail_type"] = "coordinates"
            action_log[1]["detail"] = [x, y]
            tap(adb_command_prefix, x, y)

        elif "Swipe" in action:
            try:
                coordinate1 = action.split("Swipe (")[-1].split("), (")[0].split(", ")
                coordinate2 = action.split("), (")[-1].split(")")[0].split(", ")
                x1, y1 = int(coordinate1[0]), int(coordinate1[1])
                x2, y2 = int(coordinate2[0]), int(coordinate2[1])
            except:
                print_and_log_error("ERROR: Action parameter cannot be parsed.")
                error_code = 2
                break
            action_log[0] = "Swipe"
            action_log[1][
                "detail"
            ] = f"Swipe from position ({x1}, {y1}) to position ({x2}, {y2})."
            slide(adb_command_prefix, x1, y1, x2, y2)

        elif "Type" in action:
            _action = action[4:].strip(":").strip() if action[:4] == "Type" else action
            if "(text)" not in _action and "(" in _action and ")" in _action:
                text = _action.split("(")[-1].split(")")[0]
            else:
                text = _action.split(' "')[-1].split('"')[0]
            action_log[0] = "Type"
            action_log[1]["detail"] = f'The text "{text}" has been inputted.'
            type(adb_command_prefix, text)

        elif "Back" in action:
            action_log[0] = "Back"
            action_log[1]["detail"] = "Back to the previous page."
            back(adb_command_prefix)

        elif "Home" in action:
            action_log[0] = "Home"
            action_log[1]["detail"] = "Return to home page."
            home(adb_command_prefix)

        elif "Stop" in action:
            action_log[0] = "Stop"
            action_log[1]["detail"] = "Task completed."
            task_complete = True
            break

        time.sleep(5)

        last_perception_infos = copy.deepcopy(perception_infos)
        last_screenshot_file = screenshot + "/last_screenshot.jpg"
        last_keyboard = keyboard
        if os.path.exists(last_screenshot_file):
            os.remove(last_screenshot_file)
        os.rename(screenshot_file, last_screenshot_file)

        perception_infos, width, height, api_usage = get_perception_infos(
            adb_command_prefix, screenshot_file
        )
        if perception_infos == "ERROR":
            error_code = 3
            print_and_log_error("ERROR: get perception infos failed")
            break
        shutil.rmtree(temp_file)
        os.mkdir(temp_file)

        keyboard = False
        for perception_info in perception_infos:
            if perception_info["coordinates"][1] < keyboard_height_limit:
                continue
            if "ADB Keyboard" in perception_info["text"]:
                keyboard = True
                break

        if reflection_switch:
            prompt_reflect = get_reflect_prompt(
                instruction,
                last_perception_infos,
                perception_infos,
                width,
                height,
                last_keyboard,
                keyboard,
                summary,
                action,
                add_info,
            )
            chat_reflect = init_reflect_chat()
            chat_reflect = add_response_two_image(
                "user",
                prompt_reflect,
                chat_reflect,
                [last_screenshot_file, screenshot_file],
            )

            output_reflect = inference_chat(
                chat_reflect, args.openai_api_model, API_url, token, token_storage
            )
            if output_reflect == "ERROR":
                print_and_log_error("ERROR: Inference output_reflect failed.")
                error_code = 2
            reflect = (
                output_reflect.split("### Answer ###")[-1].replace("\n", " ").strip()
            )
            chat_reflect = add_response("assistant", output_reflect, chat_reflect)
            status = "#" * 50 + " Reflcetion " + "#" * 50
            print(status)
            print(output_reflect)
            print("#" * len(status))

            if "A" in reflect:
                thought_history.append(thought)
                summary_history.append(summary)
                action_history.append(action)

                prompt_planning = get_process_prompt(
                    instruction,
                    thought_history,
                    summary_history,
                    action_history,
                    completed_requirements,
                    add_info,
                )
                chat_planning = init_memory_chat()
                chat_planning = add_response("user", prompt_planning, chat_planning)
                output_planning = inference_chat(
                    chat_planning, args.openai_api_model, API_url, token, token_storage
                )
                if output_planning == "ERROR":
                    print_and_log_error("ERROR: Inference output_planning failed.")
                    error_code = 2
                chat_planning = add_response(
                    "assistant", output_planning, chat_planning
                )
                status = "#" * 50 + " Planning " + "#" * 50
                print(status)
                print(output_planning)
                print("#" * len(status))
                completed_requirements = (
                    output_planning.split("### Completed contents ###")[-1]
                    .replace("\n", " ")
                    .strip()
                )

                error_flag = False

            elif "B" in reflect:
                error_flag = True
                reflect_back = True
                back(adb_command_prefix)

            elif "C" in reflect:
                error_flag = True

        else:
            thought_history.append(thought)
            summary_history.append(summary)
            action_history.append(action)

            prompt_planning = get_process_prompt(
                instruction,
                thought_history,
                summary_history,
                action_history,
                completed_requirements,
                add_info,
            )
            chat_planning = init_memory_chat()
            chat_planning = add_response("user", prompt_planning, chat_planning)
            output_planning = inference_chat(
                chat_planning, args.openai_api_model, API_url, token, token_storage
            )
            if output_planning == "ERROR":
                print_and_log_error("ERROR: Inference output_planning failed.")
                error_code = 2
            chat_planning = add_response("assistant", output_planning, chat_planning)
            status = "#" * 50 + " Planning " + "#" * 50
            print(status)
            print(output_planning)
            print("#" * len(status))
            completed_requirements = (
                output_planning.split("### Completed contents ###")[-1]
                .replace("\n", " ")
                .strip()
            )

        os.remove(last_screenshot_file)

        benchmark_log.append(
            {
                "step": iter,
                "response": output_action,
                "prompt_tokens": token_storage["prompt_tokens"],
                "completion_tokens": token_storage["completion_tokens"],
                "action": action_log,
                "reflect_back": reflect_back,
            }
        )
        total_prompt_tokens += token_storage["prompt_tokens"]
        total_completion_tokens += token_storage["completion_tokens"]
except Exception as e:
    print("Task finished unexpectedly")
    print_and_log_error(str(e))
    error_code = 1

end_time_exec = time.time()
elapsed_time_exec = end_time_exec - start_time_exec

benchmark_log.append(
    {
        "total_steps": iter - 1,
        "finish_signal": int(task_complete),
        "elapsed_time_initial": elapsed_time_initial,
        "elapsed_time_exec": elapsed_time_exec,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "perception_api_usage": perception_api_usage,
    }
)

with open(args.output_dir + "/log.json", "w", encoding="utf-8") as logfile:
    json.dump(benchmark_log, logfile, ensure_ascii=False)

if iter == args.max_rounds:
    error_code = 4

print(error_code)
sys.exit(error_code)
