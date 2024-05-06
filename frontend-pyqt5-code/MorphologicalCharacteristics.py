import cv2
import numpy as np
'''
可以通过以下步骤计算1和2类别所占的区域面积、周长和直径：

将图像转换为二值图像，其中1类别的像素值为255，2类别的像素值为0。
使用OpenCV的findContours函数查找图像中的轮廓。
对于每个轮廓，可以使用cv2.contourArea函数计算其面积，使用cv2.arcLength函数计算其周长。
对于每个轮廓，可以使用cv2.minEnclosingCircle函数计算其最小外接圆的半径，从而计算直径。
下面是示例代码：
'''

def calculate_characters(pred_mask, mask_original):
    pred_mask = pred_mask
    # todo 修改value值
    # 将1类别设为白色，2类别设为黑色
    # todo 这里的函数有点奇怪 1 2 都变成512，512 0矩阵了
    mask_1 = cv2.inRange(pred_mask, (250, 250, 250), (250, 250, 250))
    mask_2 = cv2.inRange(pred_mask, (125, 125, 125), (125, 125, 125))
    # cv2.imshow("Image", mask_1)  # 视杯
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imshow("Image", mask_2)  # 视盘
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 寻找轮廓
    contours_1, h_1 = cv2.findContours(mask_1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 计算1类别的面积、周长和直径
    area_1 = cv2.contourArea(contours_1[0])
    perimeter_1 = cv2.arcLength(contours_1[0], True)
    (x_1, y_1), radius_1 = cv2.minEnclosingCircle(contours_1[0])
    diameter_1 = radius_1 * 2

    print("1类别的面积：", area_1)
    print("1类别的周长：", perimeter_1)
    print("1类别的直径：", diameter_1)

    # 寻找轮廓
    contours_2, h_2 = cv2.findContours(mask_2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 计算1类别的面积、周长和直径
    area_2 = cv2.contourArea(contours_2[0])
    perimeter_2 = cv2.arcLength(contours_2[0], True)
    (x_2, y_2), radius_2 = cv2.minEnclosingCircle(contours_2[0])
    diameter_2 = radius_2 * 2

    print("2类别的面积：", area_2)
    print("2类别的周长：", perimeter_2)
    print("2类别的直径：", diameter_2)

    # 实际这个函数用来统计更合适
    counts = np.unique(mask_original, return_counts=True)  # 统计每个值的个数

    print("Counts of 1 and 2:", counts[1][1], counts[1][2])

    return diameter_1, area_1, perimeter_1, diameter_2, area_2, perimeter_2