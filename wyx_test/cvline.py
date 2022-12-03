import cv2
import numpy as np

# 画片生成 300 × 300， np.zeros()默认的变量类型是float64， 所以指定dtype为 uint8
canvas = np.zeros((300, 300, 3), dtype="uint8")

# 绿色颜色
green = (0, 255, 0)
# 画一条绿色 直线，从左上角到右下角
cv2.line(canvas, (-100, 0), (200, 300), green)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

# 红色颜色
red = (0, 0, 255)
# 画一右对角线，并指定线粗细为 3
cv2.line(canvas, (-200, 0), (0, 200), red, 3)
cv2.imshow("Canvas", canvas)

cv2.waitKey(0)
