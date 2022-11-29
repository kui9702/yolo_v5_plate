from random_plate import gen_all_plate, Draw
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


nums = 10000
draw = Draw()

results = []
with ThreadPoolExecutor(max_workers=None) as t:
    for index, i in enumerate(range(nums)):
        results.append(t.submit(gen_all_plate, draw, "/mnt/e/data/ori_plate", index, True))
    for result in tqdm.tqdm(as_completed(results), total=len(results)):
        pass