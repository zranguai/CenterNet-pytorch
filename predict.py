# 单张图片预测
from utils.centernet_predict import CenterNet
from PIL import Image


def main():
    centernet = CenterNet()  # 初始化centernet
    while True:
        img = input("input image filename:")
        try:
            image = Image.open(img)
        except:
            print("open error!")
            continue
        else:
            r_image = centernet.detect_image(image)  # 检测图片
            r_image.show()


if __name__ == '__main__':
    main()
