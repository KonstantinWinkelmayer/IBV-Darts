import cv2
import numpy as np

# Lade das Bild
image = cv2.imread('warped_image.jpg')

# Überprüfe, ob das Bild geladen wurde
if image is None:
    print("Bild konnte nicht geladen werden.")
    exit()

# Konvertiere das Bild in den HSV-Farbraum
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Definiere den Bereich für Rottöne im HSV-Farbraum
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])

# Erzeuge Masken, um nur rote Pixel zu extrahieren
mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

# Kombiniere die Masken
mask = cv2.bitwise_or(mask1, mask2)

# Wende Erosion und Dilatation an, um Rauschen zu entfernen
kernel = np.ones((5, 5), np.uint8)
mask = cv2.erode(mask, kernel, iterations=1)
mask = cv2.dilate(mask, kernel, iterations=1)

# Finde Konturen im Bild
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

largest_radius = 0
largest_circle = None

# Iteriere über alle gefundenen Konturen
for contour in contours:
    # Berechne den minimalen umschließenden Kreis
    (x, y), radius = cv2.minEnclosingCircle(contour)
    if radius > largest_radius:
        # Überprüfe, ob der Kreis eine annähernd geschlossene Form hat
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:  # Vermeidung von Division durch Null
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if 0.7 <= circularity <= 1.3:  # Anpassung der Circularitätsgrenze
            largest_radius = radius
            largest_circle = ((int(x), int(y)), int(radius))

# Wenn ein Kreis gefunden wurde
if largest_circle is not None:
    (x, y), radius = largest_circle
    # Zeichne den Kreis
    cv2.circle(image, (x, y), radius, (0, 255, 0), 2)
    # Berechne den Schwerpunkt
    centroid = (int(x), int(y))
    cv2.circle(image, centroid, 5, (0, 0, 255), -1)
    # Ausgabe der Koordinaten des Schwerpunkts
    print("Schwerpunkt Koordinaten:", centroid)
else:
    print("Kein roter Kreis gefunden.")

# Zeige das Ergebnisbild
cv2.imshow('Red Circle Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
