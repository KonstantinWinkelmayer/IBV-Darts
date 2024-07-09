from __future__ import print_function
import cv2 as cv
import argparse
import glob
from PIL import Image
import numpy as np

# Argument-Parser
parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
    OpenCV. You can process both videos, images, and GIFs.')
parser.add_argument('--input', type=str, help='Path to a single image, a sequence of images, a video, or a GIF.', default='Bilder/example.mp4')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
parser.add_argument('--speed', type=int, help='Speed of the GIF playback in milliseconds.', default=0)
parser.add_argument('--threshold_value', type=int, help='Threshold value for foreground extraction.', default=30)
args = parser.parse_args()

# Background Subtractor-Objekte erstellen
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()

# Funktion, um GIF-Frames zu extrahieren
def extract_frames_from_gif(gif_path):
    gif = Image.open(gif_path)
    frames = []
    try:
        while True:
            frame = gif.copy()
            frame = frame.convert('RGB')
            frame = np.array(frame)
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            frames.append(frame)
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    return frames

# Bilddateien aus dem angegebenen Pfad lesen
frames = []
if args.input.lower().endswith('.gif'):
    frames = extract_frames_from_gif(args.input)
elif args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Videoformate hinzufügen
    capture = cv.VideoCapture(args.input)
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frames.append(frame)
    capture.release()
else:
    image_files = sorted(glob.glob(args.input))
    if not image_files:
        print('No images found in: ' + args.input)
        exit(0)
    frames = [cv.imread(image_file) for image_file in image_files if cv.imread(image_file) is not None]

if not frames:
    print('No frames to process.')
    exit(0)

# Schleife über alle Frames
while True:
    for idx, frame in enumerate(frames):
        if frame is None:
            print('Unable to open frame:', idx)
            continue

        # Hintergrundmodell aktualisieren
        fgMask = backSub.apply(frame)

        # Anwenden des Schwellenwerts für die FG-Maske
        _, fgMask = cv.threshold(fgMask, args.threshold_value, 255, cv.THRESH_BINARY)

        # Konturen finden
        contours, _ = cv.findContours(fgMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Erstellte Kopie des Originalbildes
        display_frame = frame.copy()
        for contour in contours:
            if cv.contourArea(contour) > 500:  # Mindestgröße des Bereichs
                # Begrenzungsrechteck um den Dartpfeil
                x, y, w, h = cv.boundingRect(contour)
                cv.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.putText(display_frame, f"Position: ({x + w//2}, {y + h//2})", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Frame-Nummer anzeigen
        cv.rectangle(display_frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv.putText(display_frame, str(idx + 1), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        # Aktuelles Frame und FG-Masken anzeigen
        cv.imshow('Original Frame', frame)  # Originalbild ohne Anmerkungen
        cv.imshow('FG Mask', fgMask)
        cv.imshow('Annotated Frame', display_frame)  # Bild mit Anmerkungen

        # Tastatureingabe abwarten
        keyboard = cv.waitKey(args.speed if args.input.lower().endswith('.gif') else 30)
        if keyboard == ord('q') or keyboard == 27:  # 'q' oder ESC beendet die Schleife
            cv.destroyAllWindows()
            exit(0)

cv.destroyAllWindows()
