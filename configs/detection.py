# import modul library
from .config import NMS_THRESH
from .config import MIN_CONF
import numpy as np
import cv2

#function untuk mendeteksi orang
def detections(frame, net, ln, personIdx=0):
    
#ambil dimensi bingkai dan inisialisasi daftar hasil
    (H, W) = frame.shape[:2]
    results = []
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # inisialisasi deteksi bounding boxes, centroids, and confidence
    boxes = []
    centroids = []
    confidences = []

    # looping
    for output in layerOutputs:
        for detection in output:
            # ekstrak kelas ID dan confidence(probabilitas) dari deteksi objek saat in
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # deteksi filter dengan memastikan bahwa objek yang terdeteksi adalah orang dan bahwa confidence minimum terpenuhi
            if classID == personIdx and confidence > MIN_CONF:
                # skala koordinat bounding box  kembali relatif terhadap ukuran gambar mengembalikan koordinat pusat (x, y) diikuti dengan lebar dan tinggi kotak
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # menggunakan koordinat pusat (x,y) untuk menurunkan sudut atas dan kiri dari bounding box / kotak pembatas
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # mengupdate  bounding box coordinates, centroids dan confidences
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    idx = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

    # memastikan setidaknya ada satu deteksi
    if len(idx) > 0:
        # loop di atas indeks yang disimpan
        for i in idx.flatten():
            # ekstrak koordinat bounding box/kotak pembatas
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # memperbarui hasil dari list probabilitas prediksi orang, bounding box coordinates, dan centroid
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)

    # mengembalikan results
    return results
