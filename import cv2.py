import cv2

# Öffnen Sie die Kamera
camera = cv2.VideoCapture(1,cv2.CAP_DSHOW)  # Der Parameter 0 steht für die erste angeschlossene Kamera, 1 für die zweite usw.

# Überprüfen, ob die Kamera geöffnet wurde
if not camera.isOpened():
    print("Fehler: Kamera konnte nicht geöffnet werden")
    exit()

# Bild aufnehmen
ret, frame = camera.read()

# Überprüfen, ob das Bild aufgenommen wurde
if not ret:
    print("Fehler: Bild konnte nicht aufgenommen werden")
    exit()

# Speichern Sie das Bild auf Ihrer Festplatte
cv2.imwrite("captured_image.jpg", frame)

# Bild anzeigen
cv2.imshow("captured_image", frame)

# Warten auf eine Taste zum Schließen
cv2.waitKey(0)

# Kamera freigeben
camera.release()

# Fenster schließen
cv2.destroyAllWindows()