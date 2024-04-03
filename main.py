#Bild einlesen
import cv2

# Öffne die Kameras
cap1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap2 = cv2.VideoCapture(2, cv2.CAP_DSHOW)

# Definiere die Codec und erstelle einen VideoWriter für jede Kamera
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out1 = cv2.VideoWriter('output1.avi', fourcc, 20.0, (1280, 720))
out2 = cv2.VideoWriter('output2.avi', fourcc, 20.0, (1280, 720))

while(cap1.isOpened() and cap2.isOpened()):
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if ret1==True and ret2==True:
        # Schreibe den Frame ins Video
        out1.write(frame1)
        out2.write(frame2)

        # Zeige den Frame an
        cv2.imshow('camera1',frame1)
        cv2.imshow('camera2',frame2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Schließe alles, wenn fertig
cap1.release()
cap2.release()
out1.release()
out2.release()
cv2.destroyAllWindows()
