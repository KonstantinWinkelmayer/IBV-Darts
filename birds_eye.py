import cv2
import numpy as np

def birdseye_view(image, width, height):
    # Punkte des Quell- und Zielquadrats für die Transformation
    src_points = np.float32([[0, height], [width, height], [width, 0], [0, 0]])
    dst_points = np.float32([[width*0.2, height], [width*0.8, height], [width, 0], [0, 0]])

    # Berechne die Transformationsmatrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Führe die Perspektivtransformation durch
    birdseye = cv2.warpPerspective(image, matrix, (width, height))

    return birdseye

# Lade das Bild
image = cv2.imread('test_bild.png')
height, width = image.shape[:2]

# Erhalte die Vogelperspektive des Bildes
birdseye = birdseye_view(image, width, height)
cv2.imwrite('birds_eye.png', birdseye)

# Zeige das ursprüngliche Bild und das Bild mit Vogelperspektive an
cv2.imshow('Original', image)
cv2.imshow('Birdseye View', birdseye)
cv2.waitKey(0)
cv2.destroyAllWindows()
