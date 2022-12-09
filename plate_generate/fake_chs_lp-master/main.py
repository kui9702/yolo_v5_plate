from random_plate import gen_all_plate, Draw
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob, random, os, shutil

nums = 20000
draw = Draw()

results = []
with ThreadPoolExecutor(max_workers=None) as t:
    for index, i in enumerate(range(nums)):
        results.append(t.submit(gen_all_plate, draw, "/mnt/e/data/ori_plate", index, True))
    for result in tqdm.tqdm(as_completed(results), total=len(results)):
        pass

# results = []
# for index, i in enumerate(range(nums)):
#     gen_all_plate(draw, "/mnt/e/data/ori_plate", index, True)

# files = glob.glob(r"/mnt/d/temp_data/coco/images/train/*.jpg")
# files = random.sample(files, len(files))
# for i in files[:5000]:
#     label_path = i.replace("images", "labels").replace(".jpg",".txt")
#     if os.path.exists(label_path):
#         shutil.copyfile(i, os.path.join("/mnt/e/data/ori_plate", os.path.basename(i)))
#         shutil.copyfile(label_path, os.path.join("/mnt/e/data/ori_plate_labels", os.path.basename(label_path))) 