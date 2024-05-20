from flask import Flask, render_template, jsonify
import threading
import time
import torch
from track import detect, class_counts

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/count')
def get_count():
    global class_counts
    return jsonify(class_counts)

def start_flask():
    app.run(debug=True, use_reloader=False)

if __name__ == '__main__':
    # Flask 서버를 별도의 스레드에서 실행
    threading.Thread(target=start_flask).start()
    
    # YOLO 및 DeepSort 모델 실행
    opt = {
        "output": "output",
        "source": "0",  # 예: 웹캠을 사용할 경우
        "yolo_model": "yolov5s.pt",
        "deep_sort_model": "deep_sort/deep/checkpoint/ckpt.t7",
        "show_vid": True,
        "save_vid": False,
        "save_txt": False,
        "imgsz": 640,
        "evaluate": False,
        "half": False,
        "project": "runs/track",
        "name": "exp",
        "exist_ok": False,
        "config_deepsort": "deep_sort/configs/deep_sort.yaml",
        "conf_thres": 0.4,
        "iou_thres": 0.5,
        "max_det": 1000,
        "device": "",
        "classes": None,
        "agnostic_nms": False,
        "augment": False,
        "visualize": False,
        "dnn": False,
    }
    
    with torch.no_grad():
        detect(opt)
