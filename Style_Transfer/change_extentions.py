import cv2

image = cv2.imread('./content_image/BORI_and_SSAL.jpg')
print(image.shape)

image = cv2.resize(image, (512, 512))

cv2.imwrite('./content_image/BORI_and_SSAL.jpg', image)