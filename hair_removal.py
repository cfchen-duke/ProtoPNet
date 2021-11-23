import cv2


def remove(test, s, c):
    print("Removing hair...")
    dir = '/content/drive/Shareddrives/FYP - Skin image analysis/Skin classification-Data generation/hair removal/' + test + '/'
    src = cv2.imread(dir + test + '.jpg')
    gray_scale = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(s, (c, c))
    blackhat = cv2.morphologyEx(gray_scale, cv2.MORPH_BLACKHAT, kernel)
    ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    dst = cv2.inpaint(src, thresh2, 1, cv2.INPAINT_TELEA)
    cv2.imwrite(dir+'.jpg', dst, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
