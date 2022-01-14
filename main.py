# import modul library
from unittest import result
from xml.dom import INDEX_SIZE_ERR
from configs import config
from configs.detection import detections
from scipy.spatial import distance as dist 
import numpy as np
import argparse
import imutils
import cv2
import os

#membuat parser argumen dan parsing argumen
arp = argparse.ArgumentParser()
arp.add_argument("-i", "--input", type=str, default="", help="path to (optional) input video file")
arp.add_argument("-o", "--output", type=str, default="", help="path to (optional) output video file")
arp.add_argument("-d", "--display", type=int, default=1, help="whether or not output frame should be displayed")
args = vars(arp.parse_args())

#memuat label kelas COCO model YOLO dilatih
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# path/jalur yolov3
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

#memuat detektor objek YOLO yang dilatih pada dataset COCO
print("Loading yolo....")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

#Mengecek apakah menggunakan gpu atau tidak
if config.USE_GPU:
    #menggunakan CUDA GPU
    print("Menggunakan CUDA GPU...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

#tentukan hanya nama layer "output" yang kita butuhkan dari YOLO
layer = net.getLayerNames()
layer = [layer[i - 1] for i in net.getUnconnectedOutLayers()]

#Menginisialisasi akses video dan 
print("Mengakses video...")
vidacces = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None

#loop frame dari video
while True:
    #baca frame berikutnya dari video input
    (next, frame) = vidacces.read()
    #jika frame tidak berlanjut, maka selesai
    if not next:
        break
    
    # mengatur ukuran frame
    frame = imutils.resize(frame, width=1280)
    results = detections(frame, net, layer, personIdx=LABELS.index("person"))
    total_person = detections(frame, net, layer, personIdx=LABELS.index("person"))
    # inisialisasi nilai violation / pelanggaran
    violation = set()

    if len(results) >= 2:
        #ekstrak semua centroid dari "results" dan hitung jarak Euclidean antara semua pasangan centroid
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")
        #looping
        for i in range(0, D.shape[0]):
            for j in range(i+1, D.shape[1]):
                #memeriksa apakah jarak antara dua pasangan centroid lebih kecil daripada min distance atau jumlah piksel yang dikonfigurasi
                if D[i, j] < config.MIN_DISTANCE:
                    #perbarui set violation/pelanggaran dengan indeks pasangan centroid i dan j
                    violation.add(i)
                    violation.add(j)
    # looping
    for (i, (prob, bbox, centroid)) in enumerate(results):
        #ekstrak bounding box dan koordinat centroid, lalu inisialisasi warna 
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)
        #jika pasangan indeks ada dalam set violation/pelanggaran, maka perbarui warnanya
        if i in violation:
            color = (0, 0, 255)

        #kotak pembatas di sekitar orang tersebut dan koordinat centroid orang tersebut
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 1)

    # total orang yang terdeteksi di dalam video
    text1 = "TOTAL TERDETEKSI : {} ORANG".format(len(total_person))
    cv2.putText(frame, text1, (900, frame.shape[0] - 680), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (174, 235, 52), 3)

    # total orang melanggar social distancing/ jaga arak
    text2 = "TOTAL PELANGGARAN : {} ORANG".format(len(violation))
    cv2.putText(frame, text2, (900, frame.shape[0] - 650), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (93, 93, 240), 3)

    if args["display"] > 0:
        cv2.imshow("Output", frame)
        key = cv2.waitKey(1) & 0xFF
        #jika tombol 'x' ditekan, maka berhenti
        if key == ord("x"):
            break
    
    if args["output"] != "" and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 25, (frame.shape[1], frame.shape[0]), True)

    if writer is not None:
        print("Update output")
        writer.write(frame)


