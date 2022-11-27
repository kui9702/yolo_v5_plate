import random
import cv2
import numpy as np
import math
import os

if __name__ == "__main__":
    import black_plate
    import blue_plate
    import yellow_plate
    import green_plate
else:
    from . import black_plate
    from . import blue_plate
    from . import yellow_plate
    from . import green_plate


class Draw:

    horizontal_sight_directions = ('left', 'mid', 'right')
    vertical_sight_directions = ('up', 'mid', 'down')
    _draw = [
        black_plate.Draw(),
        blue_plate.Draw(),
        yellow_plate.Draw(),
        green_plate.Draw()
    ]
    _provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"]
    _alphabets = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    _ads = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

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
            image, matrix, size = self.left_right_transfer(image, is_left=False)
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
            image, matrix, size = self.vertical_tilt_transfer(image, is_left_high=True)
                
            image, matrix, size = self.horizontal_tilt_transfer(image, is_right_tilt=True)
        # 左上视角 或 右下视角
        elif abs(flag) == 1:
            image, matrix, size = self.vertical_tilt_transfer(image, is_left_high=False)

            image, matrix, size = self.horizontal_tilt_transfer(image, is_right_tilt=False)
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
        pts1 = np.float32([[0, 0], [0, size_src[1]], [size_src[0], 0], [size_src[0], size_src[1]]])
        # 计算图片进行投影倾斜后的位置
        interval = abs(int(math.sin((float(angle) / 180) * math.pi) * shape[0]))
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
            angle = self.angle_left_right  # self.rand_reduce(self.angle_left_right)

        shape = img.shape
        size_src = (shape[1], shape[0])
        # 源图像四个顶点坐标
        pts1 = np.float32([[0, 0], [0, size_src[1]], [size_src[0], 0], [size_src[0], size_src[1]]])
        # 计算图片进行投影倾斜后的位置
        interval = abs(int(math.sin((float(angle) / 180) * math.pi) * shape[0]))
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
        pts1 = np.float32([[0, 0], [0, size_src[1]], [size_src[0], 0], [size_src[0], size_src[1]]])
    
        # 计算图片进行上下倾斜后的距离，及形状
        interval = abs(int(math.sin((float(angle) / 180) * math.pi) * shape[1]))
        size_target = (int(math.cos((float(angle) / 180) * math.pi) * shape[1]), shape[0] + interval)
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
        pts1 = np.float32([[0, 0], [0, size_src[1]], [size_src[0], 0], [size_src[0], size_src[1]]])
        
        # 计算图片进行左右倾斜后的距离，及形状
        interval = abs(int(math.sin((float(angle) / 180) * math.pi) * shape[0]))
        size_target = (shape[1] + interval, int(math.cos((float(angle) / 180) * math.pi) * shape[0]))
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
        pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
        # 目标图像上四个顶点的坐标
        pts2 = np.float32([[self.rand_reduce(factor), self.rand_reduce(factor)],
                           [self.rand_reduce(factor), shape[0] - self.rand_reduce(factor)],
                           [shape[1] - self.rand_reduce(factor), self.rand_reduce(factor)],
                           [shape[1] - self.rand_reduce(factor), shape[0] - self.rand_reduce(factor)]])
        # 获取 3x3的投影映射/透视变换 矩阵
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        # 利用投影映射矩阵，进行透视变换
        dst = cv2.warpPerspective(img, matrix, size)
        return dst, matrix, size

    @staticmethod
    def rand_reduce(val):
        return int(np.random.random() * val)

def gen_all_plate(draw, save_dir_, index):
    plate, label = draw()
    horizontal_sight_direction = draw.horizontal_sight_directions[random.randint(0, 2)]
    vertical_sight_direction = draw.vertical_sight_directions[random.randint(0, 2)] 
    plate = draw.sight_transfer(plate, horizontal_sight_direction, vertical_sight_direction)
    plate, _, _ = draw.rand_perspective_transfer(plate)
    image_path = os.path.join(save_dir_, label+"_"+str(index)+".jpg")
    os.makedirs(save_dir_, exist_ok=True)
    cv2.imencode('.jpg', plate)[1].tofile(image_path)

if __name__ == "__main__":
    import math
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Generate a green plate.")
    parser.add_argument("--num", help="set the number of plates (default: 9)", type=int, default=9)
    args = parser.parse_args()
    draw = Draw()
    rows = math.ceil(args.num / 3)
    cols = min(args.num, 3)
    # for i in range(args.num):
    #     plate, label = draw()
    #     horizontal_sight_direction = draw.horizontal_sight_directions[random.randint(0, 2)]
    #     vertical_sight_direction = draw.vertical_sight_directions[random.randint(0, 2)] 
    #     plate = draw.sight_transfer(plate, horizontal_sight_direction, vertical_sight_direction)
    #     plate, _, _ = draw.rand_perspective_transfer(plate)


    #     print(label)
    #     plt.subplot(rows, cols, i + 1)
    #     plt.imshow(plate)
    #     plt.axis("off")
    # plt.savefig("./tenp.jpg")
    # gen_all_plate(draw, r"./", 1)
    image = cv2.imread("/mnt/d/code/python/code/202211/1.jpg")
    img1 = cv2.imread(r"/mnt/d/code/python/code/202211/YOLOv5-Lite-kile/plate_generate/license-plate-generator-master/images/11.bmp")
    img1 = cv2.resize(img1, image.shape[:2])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)  

    img1[binary.reshape(img1.shape) < 5] = img1
    img1[binary.reshape(img1.shape) > 5] = image
    
    cv2.imencode('.jpg', img1)[1].tofile("aaa.jpg")
    # contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
    # cv2.drawContours(image, contours, -1, (255, 0, 0), 2)
    