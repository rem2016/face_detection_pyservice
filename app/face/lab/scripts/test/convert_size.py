import cv2

im = cv2.imread('../../imgs/acne/4.jpg')
im = cv2.resize(im, (256, 256))
cv2.imwrite('../../imgs/acne/4s.jpg', im)