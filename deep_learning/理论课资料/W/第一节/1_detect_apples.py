import cv2
import numpy as np


def display(image_path):
    image = cv2.imread(image_path)
    cv2.imshow("show", image)

    return image

# 提取水果并标记颜色
def detect(image_path):
    img = cv2.imread(image_path)
    # 定义红色和绿色范围的下限和上限
    lower_red = np.array([0, 0, 100])
    upper_red = np.array([30, 30, 255])

    # 使用颜色范围进行红色和绿色水果的检测
    red_mask = cv2.inRange(img, lower_red, upper_red)  # 灰度图像
    cv2.imshow('red_mask', red_mask)

    # 获取轮廓
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # # 绘制轮廓
    # imag_red = cv2.drawContours(img, red_contours, -1, (0, 255, 0), 3)
    # cv2.imshow('RedContours', imag_red)
    for contour in red_contours:
        area = cv2.contourArea(contour)  # 计算轮廓面积
        if area > 300:  # 设置面积阈值，可根据实际情况调整
            x, y, w, h = cv2.boundingRect(contour)  # 把方框放置到轮廓上
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, 'Red', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow('Marked Image', img)


if __name__ == "__main__":
    img_path = 'fruit.png'  # 替换为您的图像文件路径
    # 读取图像并打开
    image = display(img_path)

    # 检测并标记水果颜色
    marked_image = detect(img_path)

    cv2.waitKey(0)  # 在规定时间内等待按键触发
    cv2.destroyAllWindows()  # 删除窗口
