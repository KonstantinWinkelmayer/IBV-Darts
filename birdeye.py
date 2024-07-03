import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_blobs(image_path):
    # Lade das Bild
    image = cv2.imread(image_path)

    # Definiere den Bereich für weiße Farben im RGB-Raum
    lower_white = np.array([200, 200, 200])  # Untere Grenze für Weiß
    upper_white = np.array([255, 255, 255])  # Obere Grenze für Weiß

    # Erstelle eine binäre Maske, in der weiße Farben im Bereich liegen
    mask = cv2.inRange(image, lower_white, upper_white)

    # Wende die Maske an, um die weißen Bereiche im Originalbild zu erhalten
    white_regions = cv2.bitwise_and(image, image, mask=mask)

    # Konvertiere das maskierte Bild in Graustufen
    gray_image = cv2.cvtColor(white_regions, cv2.COLOR_BGR2GRAY)

    # Initialisiere minCircularity und Inkrementwert
    min_circularity = 0.305
    circularity_increment = 0.001

    while True:
        # Setze die Parameter für den SimpleBlobDetector
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 200
        params.filterByCircularity = True
        params.minCircularity = min_circularity
        params.filterByConvexity = True
        params.minConvexity = 0.2
        params.filterByInertia = True
        params.minInertiaRatio = 0.01

        # Erstelle einen SimpleBlobDetector mit den Parametern
        detector = cv2.SimpleBlobDetector_create(params)

        # Erkenne Blobs im Graustufenbild
        keypoints = detector.detect(gray_image)

        # Wenn genau 4 Blobs erkannt werden, beende die Schleife
        print('anzahl gefundener Punkte:' + str(len(keypoints)) )
        print(min_circularity)

        if len(keypoints) == 4:
            break
        elif len(keypoints) < 4:
            min_circularity -= circularity_increment
        else:
            min_circularity += circularity_increment

    # Extrahiere die Mittelpunkte der erkannten Blobs
    blob_centers = [(int(keypoint.pt[0]), int(keypoint.pt[1])) for keypoint in keypoints]

    # Berechne die Homographiematrix
    dst_points = [[200 + 300, 0 + 300, 0], [0 + 300, -200 + 300, 0], [0 + 300, 200 + 300, 0], [-200 + 300, 0 + 300, 0]]
    H, _ = cv2.findHomography(np.array(blob_centers), np.array(dst_points), method=cv2.RANSAC, ransacReprojThreshold=3.0)

    # Wende die Homographie auf das Bild an
    warped_image = cv2.warpPerspective(image, H, (1000, 1000))  # Ersetze original_image, width und height entsprechend

    # Zeichne erkannte Blobs als grüne Kreise im Originalbild
    for center in keypoints:
        cv2.circle(image, (int(center.pt[0]), int(center.pt[1])), 5, (0, 255, 0), -1)

    # Zeige das Originalbild mit den Blobs an
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Weiße Blobs mit Mittelpunkten")
    plt.axis('off')
    plt.show()

    return blob_centers

# Beispielaufruf mit dem Bild "bild.jpg"
detected_blob_centers = detect_blobs("bild.jpg")