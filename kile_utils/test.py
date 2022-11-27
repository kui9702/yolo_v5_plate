from PIL import Image
import glob
import cv2, os
import tqdm

def convert(size, box):
    box = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[2]) / 2.0 - 1
    y = (box[1] + box[3]) / 2.0 - 1
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h

path = r"/mnt/d/temp_data/coco/labels/train2017"
files = glob.glob(r"/mnt/d/temp_data/coco/labels/*/*.txt")
for i in tqdm.tqdm(files):
    with open(i, "r") as r:
        text = r.read().split(" ")[-1].replace("\r", "").split(",")[:-1]
    image = cv2.imread(i.replace("labels", "images").replace(".txt", ".jpg"))
    w = image.shape[1]
    h = image.shape[0]
    x, y, w, h = convert((w, h), text)
    save_dir = os.path.dirname(i).replace("labels", "labels1")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, os.path.basename(i)), "w", encoding="utf-8") as ww:
        ww.write(f"{0} {x} {y} {w} {h}")