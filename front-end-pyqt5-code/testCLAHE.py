import cv2

# 读取图像
img = cv2.imread('../processed_data/reguge/test/image/T0001.png')

# 将图像转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用equalizeHist函数进行直方图均衡化
equalized = cv2.equalizeHist(gray)

# 创建CLAHE对象
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# 对图像进行CLAHE增强
clahe_image = clahe.apply(gray)

# 显示图像
cv2.imshow('Original', img)
cv2.imshow('Equalized', equalized)
cv2.imshow('CLAHE', clahe_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
