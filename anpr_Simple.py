import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image

# 图档
image = cv2.imread('jpgName.jpg',cv2.IMREAD_COLOR)

# 图档
image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_RGB)
plt.axis('off')
plt.show()

# 取轮廓
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 转灰階
gray = cv2.bilateralFilter(gray, 11, 17, 17)   # 模糊化，去噪
edged = cv2.Canny(gray, 30, 200)               # 取轮廓

# 图档
plt.imshow(edged, cmap='gray')
plt.axis('off')
plt.show()

# 取得等高线区域，并排序，取前10个区域
cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
cnts[0].shape

# 找第一个含四个点的等高线区域
screenCnt = None
for i, c in enumerate(cnts):
    # 计算等高线区域周长
    peri = cv2.arcLength(c, True)
    # 转为近似多边形
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    # 等高线区域维度
    print(c.shape)

    # 找第一个含四个点的多边形
    if len(approx) == 4:
        screenCnt = approx
        print(i)
        break

# 在原图上绘制多边形，框住车牌
if screenCnt is None:
    detected = 0
    print("No contour detected")
else:
    detected = 1

if detected == 1:
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
    print(f'车牌座标=\n{screenCnt}')

# 去除车牌以外的图像
mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
new_image = cv2.bitwise_and(image, image, mask=mask)

# 转为浮点数
src_pts = np.array(screenCnt, dtype=np.float32)

# 找出车牌的上下左右的座标
left = min([x[0][0] for x in src_pts])
right = max([x[0][0] for x in src_pts])
top = min([x[0][1] for x in src_pts])
bottom = max([x[0][1] for x in src_pts])

# 计算车牌宽高
width = right - left
height = bottom - top
print(f'宽度={width}, 高度={height}')

# 计算仿射的目标区域座标，须与撷取的等高线区域座标顺序相同
if src_pts[0][0][0] > src_pts[1][0][0] and src_pts[0][0][1] < src_pts[3][0][1]:
    print('起始点右上角')
    dst_pts = np.array([[width, 0], [0, 0], [0, height], [width, height]], dtype=np.float32)
elif src_pts[0][0][0] < src_pts[1][0][0] and src_pts[0][0][1] > src_pts[3][0][1]:
    print('起始点左下角')
    dst_pts = np.array([[0, height], [width, height], [width, 0], [0, 0]], dtype=np.float32)
else:
    print('起始点左上角')
    dst_pts = np.array([[0, 0], [0, height], [width, height], [width, 0]], dtype=np.float32)

# 仿射
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
Cropped = cv2.warpPerspective(gray, M, (int(width), int(height)))

# 车牌号码 OCR 辨识
text = pytesseract.image_to_string(Cropped, config='--psm 11', lang='eng')
print("车牌号码：",text)

# 显示原图及车牌
cv2.imshow('Orignal image',image)
cv2.imshow('Cropped image',Cropped)

# 车牌存档
cv2.imwrite('Cropped.jpg', Cropped)

# 按 Enter 键结束
cv2.waitKey(0)

# 关闭所有视窗
cv2.destroyAllWindows()
