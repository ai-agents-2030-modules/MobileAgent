import os
import sys
from modelscope import snapshot_download
from modelscope.pipelines import pipeline
sys.stdout.reconfigure(encoding='utf-8')

def load_cached_model(cache_path, model_name, pipeline_name, snapshot_kwargs={}):
    cache_dir = os.path.expanduser(cache_path)
    if not os.path.exists(cache_dir):
        print(f"Path {cache_dir} doesn't exist. Start to download model..")
        cache_dir = snapshot_download(model_name, **snapshot_kwargs)
    else:
        print(f"Detected path {cache_dir}. Skip download.")

    try:
        model = pipeline(pipeline_name, model=cache_dir)
        print("Loaded model successfully.")
    except Exception as e:
        print(f"Failed to load model ({cache_dir}).")
        print(f"Error: {e}")
        print("Try to download model again..")
        cache_dir = snapshot_download(model_name, **snapshot_kwargs)
        model = pipeline(pipeline_name, model=cache_dir)
    return model

def load_all_cached_models():
    from modelscope.utils.constant import Tasks
    groundingdino_model = load_cached_model(
        cache_path=r"~\.cache\modelscope\hub\AI-ModelScope\GroundingDINO",
        model_name="AI-ModelScope/GroundingDINO",
        pipeline_name="grounding-dino-task",
        snapshot_kwargs={"revision":"v1.0.0"}
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

if __name__ == '__main__':
    load_all_cached_models()