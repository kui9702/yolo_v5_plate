import numpy as np
import cv2, os
from models.experimental import attempt_load
import copy
import torch
from utils.general import non_max_suppression_landmark, xyxy2xywh
from concurrent.futures import ThreadPoolExecutor
import threading

Lock = threading.Lock()
a_finish_total = 0
a_files_total = 0

model_path = r'/mnt/d/code/python/code/202211/YOLOv5-Lite-kile/weights/best.pt'
# Load model
img_size = 320
conf_thres = 0.8
iou_thres = 0.3
device = torch.device("cpu")

model = attempt_load(model_path, "cpu")

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

def show_results(img, xywh, conf, landmarks, class_num):
    h,w,c = img.shape
    tl = 1 or round(0.2 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    for i in range(4):
        point_x = int(landmarks[2 * i] * w)
        point_y = int(landmarks[2 * i + 1] * h)
        cv2.circle(img, (point_x, point_y), tl+1, clors[0], -1)
        # print(point_x, point_y)

    tf = max(tl - 1, 1)  # font thickness
    label = str(int(class_num)) + ': ' + str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    # img[y1:y2, x1: x2] = img[0:y2-y1, 0:x2-x1]
    return img
    
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    gain = (img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    coords[:, 0] /= gain[1]
    coords[:, 1] /= gain[0]
    coords[:, 2] /= gain[1]
    coords[:, 3] /= gain[0]
    clip_coords(coords, img0_shape)
    return coords

def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    gain = (img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new

    coords[:, 0] /= gain[1]
    coords[:, 1] /= gain[0]
    coords[:, 2] /= gain[1]
    coords[:, 3] /= gain[0]
    coords[:, 4] /= gain[1]
    coords[:, 5] /= gain[0]
    coords[:, 6] /= gain[1]
    coords[:, 7] /= gain[0]
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    return coords

def warpimage(img, pts1):
    # right_bottom, left_bottom, left up , left_up
    #pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
    pts2 = np.float32([[400,200],[0,200],[0,0],[400,0]])
    h, w ,c = img.shape
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(400,200))
    return dst

def get_cor(pred, orgimg, img, image_path=None):
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
        gn_lks = torch.tensor(orgimg.shape)[[1, 0, 1, 0, 1, 0, 1, 0]].to(device)  # normalization gain landmarks
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            det[:, 5:13] = scale_coords_landmarks(img.shape[2:], det[:, 5:13], orgimg.shape).round()


            for j in range(det.size()[0]):
                xywh = (xyxy2xywh(torch.tensor(det[j, :4]).view(1, 4)) / gn).view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = (det[j, 5:13].view(1, 8) / gn_lks).view(-1).tolist()
                class_num = det[j, 13].cpu().numpy()
                orgimg = show_results(orgimg, xywh, conf, landmarks, class_num)

                h,w,c = orgimg.shape
                points = np.array(landmarks, dtype=np.float32).reshape(-1, 2)
                points[:, 0] = points[:, 0] * w
                points[:, 1] = points[:, 1] * h
                # print('points: ', points)
                dst = warpimage(orgimg, points)
                if image_path is not None:
                    # aa_path = r"/mnt/e/BaiduNetdiskDownload/crnn_test_cv"
                    aa_path = r"/mnt/d/code/python/code/202211/temp_data"
                    os.makedirs(aa_path, exist_ok=True)
                    save_path = os.path.join(aa_path, os.path.basename(image_path))
                    cv2.imwrite(save_path, dst)
                else:
                    cv2.imwrite('./result_warp_kile.jpg', dst)

            cv2.imwrite('./result_kile.jpg', orgimg)
    global a_finish_total
    with Lock:
        a_finish_total += 1
        print(f"{a_finish_total} / {a_files_total}")

def detect(image_path):
    global model
    # orgimg = cv2.imread(image_path)  # BGR
    orgimg = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    img0 = copy.deepcopy(orgimg)
    img0 = cv2.resize(img0, (img_size, img_size))
    img = img0[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416
    img = torch.from_numpy(img).to(device).float()
    img = img / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    model = model.to(device)
    pred = model(img)[0]
    pred = non_max_suppression_landmark(pred, conf_thres, iou_thres)
    get_cor(pred, orgimg, img, image_path)

def get_result(image_path):
    detect(image_path)


if __name__ == '__main__':
    # # 读取原始图片
    # img1_path = r"ori.png"
    # img1 = cv2.imread(img1_path)
    # # 获得图片的高度与宽度
    # height, weight = img1.shape[:2]

    # # 定义我们要透视变换的点，即上面图的四个点 分别是原图的 左上 右上 左下 右下， 注意需要将四个点的坐标转换成float32
    # points_ori = np.array([[13, 14], [312, 8], [24, 393], [304, 410]]).astype(np.float32)
    # # 定义我们将图片展平的点，本次展平为一张图片
    # points_transform = np.array([[0, 0], [300, 0], [0, 380], [300, 380]]).astype(np.float32)

    # # 计算得到转化矩阵  输入的参数分别是原图像的四边形坐标 变换后图片的四边形坐标
    # M = cv2.getPerspectiveTransform(points_ori, points_transform)

    # #得到透视变换的图片
    # img_ = cv2.warpPerspective(img1,M,(300,380))
	# #将图片保存即可得到2图

    # get_result(r'/mnt/e/code/python/1/202211/yolov5-car-plate-master/琼R33W7U_000000001.jpg')
    # get_result(r"/mnt/e/code/python/1/202211/yolov5-car-plate-master/data/images/0128-16_14-333&555_445&651-445&618_335&651_333&588_443&555-0_0_32_33_26_13_25-103-20.jpg")

    import glob, tqdm, random
    files = glob.glob(r"/mnt/e/BaiduNetdiskDownload/CCPD2019/ccpd_base/*.jpg")   # r"/mnt/e/BaiduNetdiskDownload/crnn_test_cv_ori/*.jpg"
    files = random.sample(files, len(files)) 
    for i in tqdm.tqdm(files):
        # if os.path.isfile(r"./result_warp_kile.jpg"):
        #     os.remove(r"./result_warp_kile.jpg")
        # if os.path.isfile(r"./result_kile.jpg"):
        #     os.remove(r"./result_kile.jpg")
        get_result(i)
    # a_files_total = len(files)
    # with ThreadPoolExecutor(max_workers=None) as t:
    #     for i in files:
    #         t.submit(get_result, i)