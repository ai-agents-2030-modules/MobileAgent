import shutil
import os
import clip
import copy
import argparse
from PIL import Image
from load_models import load_cached_model
from modelscope.utils.constant import Tasks
from MobileAgent.prompt import opreation_prompt, choose_opreation_prompt
from MobileAgent.icon_localization import det
from MobileAgent.text_localization import ocr
from MobileAgent.api import inference_chat
from MobileAgent.crop import crop, crop_for_clip, clip_for_icon
from MobileAgent.chat import init_chat, add_response, add_multiimage_response
from MobileAgent.controller import get_size, get_screenshot, tap, type, slide, back, back_to_desktop
import re
import json
import time
import sys

sys.stdout.reconfigure(encoding='utf-8')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", type=str)
    parser.add_argument("--adb_path", type=str)
    parser.add_argument("--openai_api_model", default='gpt-4-vision-preview')
    parser.add_argument("--api", type=str)
    # Add necessary arguments for benchmark
    parser.add_argument("--lang", default="ENG")
    parser.add_argument("--output_dir")
    parser.add_argument("--max_rounds", type=int, default=20)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()
    return args


def print_and_log_error(error_message):
    print(error_message)
    error_log = [{"error_message": error_message}]
    filename = args.output_dir + '/error.json'
    # Check if the file already exists
    if not os.path.exists(filename):
        # If the file does not exist, create it and write the JSON data
        with open(filename, 'w', encoding='utf-8') as logfile:
            json.dump(error_log, logfile, ensure_ascii=False)


def run(args):
    start_time_initial = time.time()
    adb_command_prefix = args.adb_path + ' -s ' + args.device
    screenshot_folder = f"./screenshot_{args.device}"
    temp_folder = f"temp_{args.device}"
    device = 'cpu'

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

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    
    observation_list, thought_list, action_list = [], [], []
    instruction = args.instruction
    struct_operation_history = init_chat(instruction)
    
    if not os.path.exists(screenshot_folder):
        os.mkdir(screenshot_folder)
    if not os.path.exists(temp_folder):
        os.mkdir(temp_folder)
    
    end_time_initial = time.time()
    elapsed_time_initial = end_time_initial - start_time_initial

    benchmark_log = []
    task_complete = False
    round_count = 0
    total_prompt_tokens, total_completion_tokens = 0, 0
    error_code = 0
    start_time_exec = time.time()
    while round_count < args.max_rounds:
        x, y = get_size(adb_command_prefix)
        get_screenshot(adb_command_prefix, screenshot_folder)
        image = screenshot_folder + "/screenshot.jpg"
        image_ori = screenshot_folder + "/screenshot.png"
        # Copy unlabelled screenshot output to benchmark results directory
        shutil.copy(src=image_ori, dst=os.path.join(args.output_dir, f'{round_count}.png'))
        round_count += 1
        temp_file = temp_folder
        iw, ih = Image.open(image).size

        if iw > ih:
            x, y = y, x
            iw, ih = ih, iw

        choose_flag = 0
        error_flag = 0
        
        operation_history = copy.deepcopy(struct_operation_history)
        operation_history = add_response("user", opreation_prompt, operation_history, image)
        
        token_storage = {"prompt_tokens": 0, "completion_tokens": 0}
        counter = 0
        while counter < 3:
            response = inference_chat(operation_history, args.api, args.openai_api_model, token_storage)
            if response == "ERROR":
                print_and_log_error("ERROR: Inference failed.")
                error_code = 3
                break
            
            try:
                observation = re.search(r"Observation:(.*?)\n", response).group(1).strip()
                thought = re.search(r"Thought:(.*?)\n", response).group(1).strip()
                action = re.search(r"Action:(.*)", response).group(1).strip()
            except:
                print("Response not formatted, retry.")
            else:
                break
            counter += 1
        
        if counter == 3:
            print_and_log_error("ERROR: Response cannot be parsed.")
            error_code = 2
        
        if error_code == 2:
            break

        observation_list.append(observation)
        thought_list.append(thought)
        action_list.append(action)
        action_log = ["", {"detail_type": "string", "detail": ""}] # second element for action details based act_name

        if "stop" in action:
            action_log[0] = "stop"
            action_log[1]["detail"] = "Task completed."
            task_complete = True
            break
        
        elif "open App" in action:
            try:
                parameter = re.search(r"\((.*?)\)", action).group(1)
            except:
                print_and_log_error("ERROR: Action parameter cannot be parsed.")
                error_code = 2
                break
            in_coordinate, out_coordinate = ocr(image_ori, parameter, ocr_detection, ocr_recognition, iw, ih)
            action_log[0] = f"open App \"({parameter})\""
            
            if len(in_coordinate) == 0:
                action_log[1]["detail"] = f"Invalid action: No App named \"{parameter}\"."
                error_prompt = f"No App named {parameter}."
                error_flag = 1
            else:
                tap_coordinate = [(in_coordinate[0][0]+in_coordinate[0][2])/2, (in_coordinate[0][1]+in_coordinate[0][3])/2]
                tap_coordinate = [round(tap_coordinate[0]/iw, 2), round(tap_coordinate[1]/ih, 2)]
                action_log[1]["detail_type"] = "coordinates"
                action_log[1]["detail"] = [int(tap_coordinate[0] * x),int(tap_coordinate[1]-round(50/y, 2) * y)]
                tap(adb_command_prefix, tap_coordinate[0], tap_coordinate[1]-round(50/y, 2), x, y)
        
        elif "click text" in action:
            
            choose_chat = init_chat(instruction)
            choose_chat = add_response("user", choose_opreation_prompt, choose_chat, image)
            choose_chat = add_response("assistant", action, choose_chat)

            try:
                parameter = re.search(r"\((.*?)\)", action).group(1)
            except:
                print_and_log_error("ERROR: Action parameter cannot be parsed.")
                error_code = 2
                break
            in_coordinate, out_coordinate = ocr(image_ori, parameter, ocr_detection, ocr_recognition, iw, ih)
            action_log[0] = f"click text \"{parameter}\""
                
            if len(out_coordinate) == 0:
                action_log[1]["detail"] = f"Invalid action: The text \"{parameter}\" is not detected in the screenshot."
                error_prompt = f"Failed to execute action click text ({parameter}). The text \"{parameter}\" is not detected in the screenshot."
                error_flag = 1
            elif len(out_coordinate) > 4:
                action_log[1]["detail"] = f"Invalid action: There are too many text \"{parameter}\" in the screenshot."
                error_prompt = f"Failed to execute action click text ({parameter}). There are too many text \"{parameter}\" in the screenshot."
                error_flag = 1
            
            elif len(out_coordinate) == 1:
                tap_coordinate = [(in_coordinate[0][0]+in_coordinate[0][2])/2, (in_coordinate[0][1]+in_coordinate[0][3])/2]
                tap_coordinate = [round(tap_coordinate[0]/iw, 2), round(tap_coordinate[1]/ih, 2)]
                action_log[1]["detail_type"] = "coordinates"
                action_log[1]["detail"] = [int(tap_coordinate[0] * x),int(tap_coordinate[1] * y)]
                tap(adb_command_prefix, tap_coordinate[0], tap_coordinate[1], x, y)
            
            else:
                hash = {}
                for i, (td, box) in enumerate(zip(in_coordinate, out_coordinate)):
                    crop(image, box, i+1, temp_folder, td)
                    hash[i+1] = td

                images = []
                temp_file = temp_folder
                for i in range(len(hash.keys())):
                    crop_image = f"{i+1}.jpg"
                    images.append(os.path.join(temp_file, crop_image))
                
                ocr_prompt = f"The {str(len(out_coordinate))} red boxes are numbered 1 through {str(len(out_coordinate))}. Which red box with \"{parameter}\" do you want to click on? Please output just one number from 1 to {str(len(out_coordinate))}, such as 1, 2......"
                choose_chat = add_multiimage_response("user", ocr_prompt, choose_chat, images)
                choose_response = inference_chat(choose_chat, args.api, args.openai_api_model, token_storage)
                
                final_box = hash[int(choose_response)]
                tap_coordinate = [(final_box[0]+final_box[2])/2, (final_box[1]+final_box[3])/2]
                tap_coordinate = [round(tap_coordinate[0]/iw, 2), round(tap_coordinate[1]/ih, 2)]
                action_log[1]["detail_type"] = "coordinates"
                action_log[1]["detail"] = [int(tap_coordinate[0] * x),int(tap_coordinate[1] * y)]
                tap(adb_command_prefix, tap_coordinate[0], tap_coordinate[1], x, y)
                
                choose_flag = 1
                choose_user = ocr_prompt
                choose_images = images
                choose_response = choose_response
        
        elif "click icon" in action:
            
            choose_chat = init_chat(instruction)
            choose_chat = add_response("user", choose_opreation_prompt, choose_chat, image)
            choose_chat = add_response("assistant", action, choose_chat)

            try:
                parameter = re.search(r"\((.*?)\)", action).group(1)
            except:
                print_and_log_error("ERROR: Action parameter cannot be parsed.")
                error_code = 2
                break
            parameter1, parameter2 = parameter.split(',')[0].strip(), parameter.split(',')[1].strip()
            in_coordinate, out_coordinate = det(image, "icon", groundingdino_model)
            action_log[0] = f"click icon ({parameter1}, {parameter2})"
            action_log[1]["detail_type"] = "coordinates"
            
            if len(out_coordinate) == 1:
                tap_coordinate = [(in_coordinate[0][0]+in_coordinate[0][2])/2, (in_coordinate[0][1]+in_coordinate[0][3])/2]
                tap_coordinate = [round(tap_coordinate[0]/iw, 2), round(tap_coordinate[1]/ih, 2)]
                action_log[1]["detail"] = [int(tap_coordinate[0] * x),int(tap_coordinate[1] * y)]
                tap(adb_command_prefix, tap_coordinate[0], tap_coordinate[1], x, y)
                
            else:
                temp_file = temp_folder
                hash = []
                clip_filter = []
                for i, (td, box) in enumerate(zip(in_coordinate, out_coordinate)):
                    if crop_for_clip(image, td, i+1, temp_folder, parameter2):
                        hash.append(td)
                        crop_image = f"{i+1}.jpg"
                        clip_filter.append(os.path.join(temp_file, crop_image))
                    
                clip_filter = clip_for_icon(clip_model, clip_preprocess, clip_filter, parameter1)
                final_box = hash[clip_filter]
                tap_coordinate = [(final_box[0]+final_box[2])/2, (final_box[1]+final_box[3])/2]
                tap_coordinate = [round(tap_coordinate[0]/iw, 2), round(tap_coordinate[1]/ih, 2)]
                action_log[1]["detail"] = [int(tap_coordinate[0] * x),int(tap_coordinate[1] * y)]
                tap(adb_command_prefix, tap_coordinate[0], tap_coordinate[1], x, y)
                    
        elif "page" in action:
            if "down" in action:
                action_log[0] = "page down"
            elif "up" in action:
                action_log[0] = "page up"
            else:
                action_log[1]["detail"] = "Invalid action: cannot be executed."
            slide(adb_command_prefix, action, x, y)
        
        elif "type" in action:
            try:
                text = re.search(r"\((.*?)\)", action).group(1)
            except:
                print_and_log_error("ERROR: Action parameter cannot be parsed.")
                error_code = 2
                break
            action_log[0] = "type"
            action_log[1]["detail"] = f"The text \"{text}\" has been inputted."
            type(adb_command_prefix, text)
        
        elif "back" in action:
            action_log[0] = "back"
            action_log[1]["detail"] = "Back to the previous page."
            back(adb_command_prefix)

        elif "exit" in action:
            action_log[0] = "exit"
            action_log[1]["detail"] = "Exit the app and go back to the desktop."
            back_to_desktop(adb_command_prefix)

        else:
            action_log[1]["detail"] = "No action matched."
            error_prompt = "Please respond strictly to the output format!"
        
        struct_operation_history = add_response("user", "This is the current screenshot. Please give me your action.", struct_operation_history, image)
        struct_operation_history = add_response("assistant", action, struct_operation_history)
        
        if error_flag == 0:
            if choose_flag == 1:
                struct_operation_history = add_multiimage_response("user", choose_user, struct_operation_history, choose_images)
                struct_operation_history = add_response("assistant", choose_response, struct_operation_history)
        else:
            struct_operation_history = add_response("user", error_prompt, struct_operation_history, image)
            struct_operation_history = add_response("assistant", "I will try again with another action or parameter.", struct_operation_history)
        
        shutil.rmtree(temp_file)
        os.mkdir(temp_file)

        benchmark_log.append({"step": round_count, "response": response, "prompt_tokens": token_storage["prompt_tokens"], "completion_tokens": token_storage["completion_tokens"], "action": action_log})
        total_prompt_tokens += token_storage["prompt_tokens"]
        total_completion_tokens += token_storage["completion_tokens"]

    end_time_exec = time.time()
    elapsed_time_exec = end_time_exec - start_time_exec

    benchmark_log.append({
        "total_steps": round_count - 1, "finish_signal": int(task_complete),
        "elapsed_time_initial": elapsed_time_initial, "elapsed_time_exec": elapsed_time_exec,
        "total_prompt_tokens": total_prompt_tokens, "total_completion_tokens": total_completion_tokens
    })

    with open(args.output_dir + '/log.json', "w", encoding='utf-8') as logfile:
        json.dump(benchmark_log, logfile, ensure_ascii=False)

    if round_count == args.max_rounds:
        error_code = 4

    return error_code

if __name__ == "__main__":
    args = get_args()
    try:
        error_code = run(args)
    except Exception as err:
        print("Task finished unexpectedly")
        print_and_log_error(str(err))
        error_code = 1
    sys.exit(error_code)
