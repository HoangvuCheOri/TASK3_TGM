#Import và load cascade từ thư viện OpenCV
import cv2
from pathlib import Path
from natsort import natsorted 
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#Đọc ảnh và chỉnh ảnh sang màu xám (Viola-Jones)
img_lib = Path("img")                                 #Trỏ tới thư mục "img"
images = natsorted(img_lib.glob("*"))                      #Lấy hết ảnh trong thư mục
for img_path in images:                               #Xét từng ảnh
    print("Đang detect:", img_path.name)
    image = cv2.imread(str(img_path))                 #chuyển path -> string (openCV)
    if image is None:
        print("Không đọc được ảnh, bỏ qua.")
        continue
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    #Đổi màu
    gray = cv2.equalizeHist(gray)                     #Tương phản dễ đọc

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)     #Detect nhiều mặt với 5 lần xét, scale factor 1.1 (10%)
    for (x, y, w, h) in faces:
     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2)   #Tạo bounding box hình vuông màu vàng

#Kết quả thu được
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Result", image)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

 
