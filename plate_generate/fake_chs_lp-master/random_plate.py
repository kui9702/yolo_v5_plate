import green_plate
import yellow_plate
import blue_plate
import black_plate
import random
import cv2
import numpy as np
import math
import os


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


def points2yolo(size, points):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    rbx = points[0] * dw
    rby = points[1] * dh
    lbx = points[2] * dw
    lby = points[3] * dh
    ltx = points[4] * dw
    lty = points[5] * dh
    rtx = points[6] * dw
    rty = points[7] * dh
    return rbx, rby, lbx, lby, ltx, lty, rtx, rty


class Draw:

    horizontal_sight_directions = ('left', 'mid', 'right')
    vertical_sight_directions = ('up', 'mid', 'down')
    _draw = [
        # black_plate.Draw(),
        blue_plate.Draw(),
        yellow_plate.Draw(),
        green_plate.Draw()
    ]
    _provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽",
                  "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"]
    _alphabets = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L",
                  "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    _ads = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R",
            "S", "T", "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    angle_left_right = 5
    angle_horizontal = 15
    angle_vertical = 15
    angle_up_down = 10
    factor = 10

    def __call__(self):
        draw = random.choice(self._draw)
        candidates = [self._provinces, self._alphabets]
        if type(draw) == green_plate.Draw:
            candidates += [self._ads] * 6
            label = "".join([random.choice(c) for c in candidates])
            return draw(label, random.randint(0, 1)), label
        elif type(draw) == black_plate.Draw:
            if random.random() < 0.5:
                candidates += [self._ads] * 4
                candidates += [["港", "澳"]]
            else:
                candidates += [self._ads] * 5
            label = "".join([random.choice(c) for c in candidates])
            return draw(label), label
        elif type(draw) == yellow_plate.Draw:
            if random.random() < 0.5:
                candidates += [self._ads] * 4
                candidates += [["学"]]
            else:
                candidates += [self._ads] * 5
            label = "".join([random.choice(c) for c in candidates])
            return draw(label), label
        else:
            candidates += [self._ads] * 5
            label = "".join([random.choice(c) for c in candidates])
            return draw(label), label

    def sight_transfer(self, image, horizontal_sight_direction, vertical_sight_direction):
        """ 对图片进行视角变换
        :param images: 图片列表
        :param horizontal_sight_direction: 水平视角变换方向
        :param vertical_sight_direction: 垂直视角变换方向
        :return:
        """
        flag = 0
        # 左右视角
        if horizontal_sight_direction == 'left':
            flag += 1
            image, matrix, size = self.left_right_transfer(image, is_left=True)
        elif horizontal_sight_direction == 'right':
            flag -= 1
            image, matrix, size = self.left_right_transfer(
                image, is_left=False)
        else:
            pass
        # 上下视角
        if vertical_sight_direction == 'down':
            flag += 1
            image, matrix, size = self.up_down_transfer(image, is_down=True)
        elif vertical_sight_direction == 'up':
            flag -= 1
            image, matrix, size = self.up_down_transfer(image, is_down=False)
        else:
            pass

        # 左下视角 或 右上视角
        if abs(flag) == 2:
            image, matrix, size = self.vertical_tilt_transfer(
                image, is_left_high=True)

            image, matrix, size = self.horizontal_tilt_transfer(
                image, is_right_tilt=True)
        # 左上视角 或 右下视角
        elif abs(flag) == 1:
            image, matrix, size = self.vertical_tilt_transfer(
                image, is_left_high=False)

            image, matrix, size = self.horizontal_tilt_transfer(
                image, is_right_tilt=False)
        else:
            pass

        return image

    def up_down_transfer(self, img, is_down=True, angle=None):
        """ 上下视角，默认下视角
        :param img: 正面视角原始图片
        :param is_down: 是否下视角
        :param angle: 角度
        :return:
        """
        if angle is None:
            angle = self.rand_reduce(self.angle_up_down)

        shape = img.shape
        size_src = (shape[1], shape[0])
        # 源图像四个顶点坐标
        pts1 = np.float32([[0, 0], [0, size_src[1]], [
                          size_src[0], 0], [size_src[0], size_src[1]]])
        # 计算图片进行投影倾斜后的位置
        interval = abs(
            int(math.sin((float(angle) / 180) * math.pi) * shape[0]))
        # 目标图像上四个顶点的坐标
        if is_down:
            pts2 = np.float32([[interval, 0], [0, size_src[1]],
                               [size_src[0] - interval, 0], [size_src[0], size_src[1]]])
        else:
            pts2 = np.float32([[0, 0], [interval, size_src[1]],
                               [size_src[0], 0], [size_src[0] - interval, size_src[1]]])
        # 获取 3x3的投影映射/透视变换 矩阵
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, matrix, size_src)
        return dst, matrix, size_src

    def left_right_transfer(self, img, is_left=True, angle=None):
        """ 左右视角，默认左视角
        :param img: 正面视角原始图片
        :param is_left: 是否左视角
        :param angle: 角度
        :return:
        """
        if angle is None:
            # self.rand_reduce(self.angle_left_right)
            angle = self.angle_left_right

        shape = img.shape
        size_src = (shape[1], shape[0])
        # 源图像四个顶点坐标
        pts1 = np.float32([[0, 0], [0, size_src[1]], [
                          size_src[0], 0], [size_src[0], size_src[1]]])
        # 计算图片进行投影倾斜后的位置
        interval = abs(
            int(math.sin((float(angle) / 180) * math.pi) * shape[0]))
        # 目标图像上四个顶点的坐标
        if is_left:
            pts2 = np.float32([[0, 0], [0, size_src[1]],
                               [size_src[0], interval], [size_src[0], size_src[1] - interval]])
        else:
            pts2 = np.float32([[0, interval], [0, size_src[1] - interval],
                               [size_src[0], 0], [size_src[0], size_src[1]]])
        # 获取 3x3的投影映射/透视变换 矩阵
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, matrix, size_src)
        return dst, matrix, size_src

    def vertical_tilt_transfer(self, img, is_left_high=True):
        """ 添加按照指定角度进行垂直倾斜(上倾斜或下倾斜，最大倾斜角度self.angle_vertical一半）
        :param img: 输入图像的numpy
        :param is_left_high: 图片投影的倾斜角度，左边是否相对右边高
        """
        angle = self.rand_reduce(self.angle_vertical)

        shape = img.shape
        size_src = [shape[1], shape[0]]
        # 源图像四个顶点坐标
        pts1 = np.float32([[0, 0], [0, size_src[1]], [
                          size_src[0], 0], [size_src[0], size_src[1]]])

        # 计算图片进行上下倾斜后的距离，及形状
        interval = abs(
            int(math.sin((float(angle) / 180) * math.pi) * shape[1]))
        size_target = (int(math.cos((float(angle) / 180) * math.pi)
                       * shape[1]), shape[0] + interval)
        # 目标图像上四个顶点的坐标
        if is_left_high:
            pts2 = np.float32([[0, 0], [0, size_target[1] - interval],
                               [size_target[0], interval], [size_target[0], size_target[1]]])
        else:
            pts2 = np.float32([[0, interval], [0, size_target[1]],
                               [size_target[0], 0], [size_target[0], size_target[1] - interval]])

        # 获取 3x3的投影映射/透视变换 矩阵
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, matrix, size_target)
        return dst, matrix, size_target

    def horizontal_tilt_transfer(self, img, is_right_tilt=True):
        """ 添加按照指定角度进行水平倾斜(右倾斜或左倾斜，最大倾斜角度self.angle_horizontal一半）
        :param img: 输入图像的numpy
        :param is_right_tilt: 图片投影的倾斜方向（右倾，左倾）
        """
        angle = self.rand_reduce(self.angle_horizontal)

        shape = img.shape
        size_src = [shape[1], shape[0]]
        # 源图像四个顶点坐标
        pts1 = np.float32([[0, 0], [0, size_src[1]], [
                          size_src[0], 0], [size_src[0], size_src[1]]])

        # 计算图片进行左右倾斜后的距离，及形状
        interval = abs(
            int(math.sin((float(angle) / 180) * math.pi) * shape[0]))
        size_target = (
            shape[1] + interval, int(math.cos((float(angle) / 180) * math.pi) * shape[0]))
        # 目标图像上四个顶点的坐标
        if is_right_tilt:
            pts2 = np.float32([[interval, 0], [0, size_target[1]],
                               [size_target[0], 0], [size_target[0] - interval, size_target[1]]])
        else:
            pts2 = np.float32([[0, 0], [interval, size_target[1]],
                               [size_target[0] - interval, 0], [size_target[0], size_target[1]]])

        # 获取 3x3的投影映射/透视变换 矩阵
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, matrix, size_target)
        return dst, matrix, size_target

    def rand_perspective_transfer(self, img, factor=None, size=None):
        """ 添加投影映射畸变
        :param img: 输入图像的numpy
        :param factor: 畸变的参数
        :param size: 图片的目标尺寸，默认维持不变
        """
        if factor is None:
            factor = self.factor
        if size is None:
            size = (img.shape[1], img.shape[0])
        shape = size
        # 源图像四个顶点坐标
        pts1 = np.float32(
            [[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
        # 目标图像上四个顶点的坐标
        pts2 = np.float32([[self.rand_reduce(factor), self.rand_reduce(factor)],
                           [self.rand_reduce(factor), shape[0] -
                            self.rand_reduce(factor)],
                           [shape[1] -
                               self.rand_reduce(factor), self.rand_reduce(factor)],
                           [shape[1] - self.rand_reduce(factor), shape[0] - self.rand_reduce(factor)]])
        # 获取 3x3的投影映射/透视变换 矩阵
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        # 利用投影映射矩阵，进行透视变换
        dst = cv2.warpPerspective(img, matrix, size)
        return dst, matrix, size

    @staticmethod
    def rand_reduce(val):
        return int(np.random.random() * val)

    def get_ture_cor(self, points, x1, y1, w, h):  # return rb lb lt rt
        if len(points) != 4:
            return None, None
        else:
            points = np.array(points)
            points = points.reshape((4, -1))
        xmin, xmax = np.min(points[:, 0]), np.max(points[:, 0])
        ymin, ymax = np.min(points[:, 1]), np.max(points[:, 1])
        for i in points:
            if int(i[0]) + w / 5 * 3 < xmax:
                if i[1] + h / 5 * 3 < ymax:
                    ltx = i[0]
                    lty = i[1]
                else:
                    lbx = i[0]
                    lby = i[1]
            else:
                if i[1] + h / 5 * 3 < ymax:
                    rtx = i[0]
                    rty = i[1]
                else:
                    rbx = i[0]
                    rby = i[1]
        try:
            return [rbx+x1, rby+y1, lbx+x1, lby+y1, ltx+x1, lty+y1, rtx+x1, rty+y1], [xmin+x1, ymin+y1, xmax+x1, ymax+y1]
        except:
            return None, None

    def add_pure_image(self, env, img, get_cor=False):
        if env is None:
            env = cv2.imread(
                r"/home/kile/files/yolo_v5_plate/plate_generate/fake_chs_lp-master/background/result_kile.jpg")
            env = cv2.resize(env, (random.randint(
                img.shape[0]*10, img.shape[0]*15), random.randint(img.shape[0]*10, img.shape[0]*15)))
        x1 = random.randint(0, env.shape[1] - img.shape[1])
        y1 = random.randint(0, env.shape[0] - img.shape[0])
        roi = env[y1:img.shape[0]+y1, x1:x1+img.shape[1]]

        # 创建掩膜
        img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("./temp.jpg", img2gray)
        a = img2gray.copy()
        a[a<65]=255
        a[a>90]=255
        a[a!=255] = True
        a[a==255] = False
        a = np.array(a, dtype=bool)
        # cv2.imwrite("./temp.jpg", a)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        # cv2.imwrite("./temp.jpg", mask)
        mask_inv = cv2.bitwise_not(mask)
        # cv2.imwrite("./temp.jpg", mask_inv)

        # 保留除logo外的背景
        img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        # cv2.imwrite("./temp.jpg", img1_bg)
        dst = cv2.add(img1_bg, img)  # 进行融合
        # cv2.imwrite("./temp.jpg", dst)
        # dst = cv2.add(dst, a)
        dst[a]=[0,0,0]
        cv2.imwrite("./temp.jpg", dst)
        # dst = cv2.add(dst, a)
        env[y1:img.shape[0]+y1, x1:x1+img.shape[1]] = dst  # 融合后放在原图上

        # 添加 四个顶点坐标
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # imgaa = env.copy()
        cnt_len = cv2.arcLength(contours[0], True)
        cnt = cv2.approxPolyDP(contours[0], 0.02*cnt_len, True)
        points = []
        points, bbox = self.get_ture_cor(cnt, x1, y1, img.shape[1], img.shape[0])
        if not get_cor:
            return env
        else:
            return env, bbox, points


def show_results(img, xyxy, landmarks):
    imgcopy = img.copy()
    cv2.rectangle(imgcopy, (int(xyxy[0]), int(xyxy[1])), (int(
        xyxy[2]), int(xyxy[3])), (0, 255, 0), lineType=cv2.LINE_AA)

    clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
             (255, 255, 0), (0, 255, 255)]

    for i in range(0, 8, 2):
        point_x = int(landmarks[i])
        point_y = int(landmarks[i+1])
        cv2.circle(imgcopy, (point_x, point_y), 2, clors[0], -1)
    aa_path = r"/mnt/e/data/temp_data"
    os.makedirs(aa_path, exist_ok=True)
    save_path = os.path.join(aa_path, str(random.randint(0, 10000))+".jpg")
    cv2.imwrite(save_path, imgcopy)


def gen_all_plate(draw, save_dir_, index, get_cor=False):
    plate, label = draw()
    plate = cv2.cvtColor(plate, cv2.COLOR_RGB2BGR)
    horizontal_sight_direction = draw.horizontal_sight_directions[random.randint(
        0, 2)]
    vertical_sight_direction = draw.vertical_sight_directions[random.randint(
        0, 2)]
    plate = draw.sight_transfer(
        plate, horizontal_sight_direction, vertical_sight_direction)
    plate, _, _ = draw.rand_perspective_transfer(plate)
    image_path = os.path.join(save_dir_, label+"_"+str(index)+".jpg")
    if get_cor:
        plate, box, points = draw.add_pure_image(None, plate, get_cor)
        if points == None:
            return
        # show_results(plate, box, points)
        os.makedirs(save_dir_+"_labels", exist_ok=True)
        label_path = os.path.join(
            save_dir_+"_labels", label+"_"+str(index)+".txt")
        x, y, w, h = convert((plate.shape[1], plate.shape[0]), box)
        rbx, rby, lbx, lby, ltx, lty, rtx, rty = points2yolo(
            (plate.shape[1], plate.shape[0]), points)
        with open(label_path, "w", encoding="utf-8") as ww:
            ww.write(
                f"{0} {x} {y} {w} {h} {rbx} {rby} {lbx} {lby} {ltx} {lty} {rtx} {rty}")
    else:
        plate = draw.add_pure_image(None, plate)
    os.makedirs(save_dir_, exist_ok=True)
    cv2.imencode('.jpg', plate)[1].tofile(image_path)
    # cv2.imwrite(image_path, plate)
    
    
    #验证数据是否正确
    # import numpy as np
    # # points = np.array(points).reshape(2,-1)
    # tl = 1 or round(0.2 * (h + w) / 2) + 1  # line/font thickness
    # for i in range(4):
    #     point_x = int(points[2 * i])
    #     point_y = int(points[2 * i + 1])
    #     cv2.circle(plate, (point_x, point_y), tl+1, (255,0,0), -1)
    # aa_path = r"/mnt/e/data/temp_data"
    # os.makedirs(aa_path, exist_ok=True)
    # save_path = os.path.join(aa_path, os.path.basename(image_path))
    # cv2.imencode('.jpg', plate)[1].tofile(save_path)
