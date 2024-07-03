import cv2

# Öffne die Kamera
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Lese ein Bild von der Kamera
ret, frame = cap.read()

if ret:
    # Speichere das aufgenommene Bild
    cv2.imwrite('bild.jpg', frame)

    # Zeige das aufgenommene Bild an
    cv2.imshow('Kamera', frame)
    
    # Warte bis eine Taste gedrückt wird und schließe dann das Fenster
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Fehler beim Aufnehmen des Bildes")

# Schließe die Kamera
cap.release()
