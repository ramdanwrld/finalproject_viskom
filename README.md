# Sistem Deteksi Social Distancing (Jaga Jarak) di Tempat Umum dengan OpenCV menggunakan Algoritma YOLO

Anggota:

Azmi Aulia Rahman (1301184086)

Nuraena Ramdani (1301180373)

Langkah-Langkah:
1. Download terlebih dahulu file di repository github ini dan file yang ada di google drive yang berisi file yolov3.weights dan video.mp4 rekaman yang dibutuhkan untuk menjalan program ini.
Atau jika mau lihat langsung hasilnya ada output file yang sudah di running yang bernama output.avi .
2. Link google drive : https://drive.google.com/drive/folders/123kBaMnIIeRHnCSZJwVK9h8s5kyYFXso?usp=sharing 
(maaf karena ukuran file yolov3.weights ini cukup besar sehingga tidak bisa diupload ke github melebihi 100mb)
2. masukkan video.mp4 yang telah didownload ke dalam satu folder
2. masukkan yolov3.weights ke dalam folder yolo-coco
3. buka file "untuk run program.txt" dan copy text 
4. selanjutnya buka cmd di folder tersebut dan paste atau ketik
python main.py --input video.mp4 --output output_video.avi --display 1
5. dan akan menampilkan proses frame deteksi social distancing atau jaga jarak
6. jika sudah selesai maka akan menghasilkan output.avi yang dapat dibuka dan dilihat hasilnya dari video rekaman tadi menjadi terdapat deteksi social distancing.