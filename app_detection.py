import os  # obsługa systemu plików
import sys  # obsługa funkcji systemowych
import cv2  # biblioteka do przetwarzania obrazu i wideo
import yaml  # obsługa plików YAML - config modeli
import logging.config  # konfiguracja logowania
import numpy as np
from tkinter import filedialog  # okno dialogowe do wyboru plików wideo
import datetime

from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader # klasy do łądowania detektora twarzy
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler # klasy do obsługi detektora twarzy

logging.config.fileConfig("config/logging.conf") # konfiguracja logowania z pliku konfiguracyjnego logging.conf
logger = logging.getLogger('api')  # Ustawiamy logger o nazwie 'api'

# wczytanie konfiguracji z pliku YAML
with open('config/model_conf.yaml') as f:
    model_conf = yaml.safe_load(f)  # załadowanie konfiguracji


def load_model():

    model_path = 'models'  # path do folderu z modelami
    scene = 'non-mask'  # scena bez maski
    model_category = 'face_detection'  # kategoria modelu - detekcja twarzy
    model_name = model_conf[scene][model_category]  # nazwa modelu z konfiguracji

    logger.info('Loading the face detection model...')  # informacja o ładowaniu modelu w loggerze
    try:
        # Tworzymy obiekt loadera modelu i wczytujemy model oraz jego konfigurację
        model_loader = FaceDetModelLoader(model_path, model_category, model_name) # tworzenie obiektu do ładowania modelu detekcji
        model, conf = model_loader.load_model() # wczytanie modelu i jego konfiguracji - model, conf

        logger.info('Model loaded successfully!')  # informacja o sukcesie wczytania tego modelu do loggera

        return FaceDetModelHandler(model, 'cuda:0', conf)  # zwracamy handler do obsługi modelu (na GPU)

    except Exception as e:
        logger.error('Failed to load model:', exc_info=True)  # logujemy błąd do loggera, jeśli coś poszło nie tak
        sys.exit(-1)  # zakończenie programu


def detect_and_save(cap, entity_name):
    detection_handler = load_model() # załadowanie modelu detekcji
    output_dir = 'collected_images' # ścieżka zapisu zdjęć
    os.makedirs(output_dir, exist_ok=True) # upewnienie się, że folder istnieje

    show_blue_box = False # włączenie opcji wyświetlania niebieskiej ramki po zapisie
    blue_box_end_time = None
    blue_box_coords = None

    while cap.isOpened():
        ret, frame = cap.read() # pobranie klatki
        if not ret:
            break

        raw_frame = frame.copy() # zachowanie czystej klatki bez obróbki

        try:
            detections = detection_handler.inference_on_image(frame) # wykonanie detekcji twarzy
        except Exception as e:
            logging.error('Face detection failed:', exc_info=True)
            detections = []

        current_time = datetime.datetime.now()

        # rysowanie zielonych ramek wokół wykrytych twarzy
        for box in detections:
            box = list(map(int, box))
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        # rysowanie niebieskiej ramki po zapisaniu zdjęcia
        if show_blue_box and blue_box_end_time and current_time < blue_box_end_time and blue_box_coords is not None:
            x1, y1, x2, y2 = blue_box_coords
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imshow("Dodawanie wzorca (R = zapisz, Q = wyjdz)", frame) # wyświetlenie klatki
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            break # zakończenie działania

        elif (key == ord('r') or key == ord('R')) and len(detections) > 0:
            best_box = list(map(int, detections[0])) # wybranie pierwszej wykrytej twarzy
            now = datetime.datetime.now()  # pobranie bieżącego czasu
            time_str = now.strftime("%H_%M_%d_%m_%Y") # czas aktualny w formacie godzina_minuta_dzień_miesiąc_rok
            filename = os.path.join(output_dir, f"{entity_name}_{time_str}.jpg") # nazwa pliku złożona z nazwy pliku i daty
            cv2.imwrite(filename, raw_frame) # zapisanie czystej klatki jako plik
            logging.info(f"Zapisano zdjęcie jako {filename}")

            # ustawienie niebieskiej ramki na 0,2 sekundy
            show_blue_box = True
            blue_box_end_time = now + datetime.timedelta(seconds=0.2)
            blue_box_coords = (best_box[0], best_box[1], best_box[2], best_box[3])

    cap.release() # zwolnienie zasobów kamery
    cv2.destroyAllWindows() # zamknięcie wszystkich okien OpenCV


# Funkcja obsługująca detekcję za pomocą kamery internetowej
def detect_camera(entity_name):

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logger.error("Failed to open camera.")
        return # jeśli kamera nie została otwarta poprawnie, logujemy błąd i kończymy funkcję

    detect_and_save(cap, entity_name) # przekazujemy uchwyt kamery do funkcji detekcji i zapisu


def detect_video(entity_name):
    video_path = filedialog.askopenfilename(title="Wybierz plik wideo.",
                                            filetypes=[("Video Files", "*.mp4;*.avi;*.mkv")]) # okno dialogowe wyboru pliku wideo

    if not video_path:
        logger.warning("No video file selected.")
        return # jeśli użytkownik nie wybierze pliku, ostrzeżenie zapisywane w loggerze i kończymy funkcję

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Failed to open video file.")
        return # jeśli nie udało się otworzyć pliku wideo, błąd zapisywany w loggerze i kończymy funkcję

    detect_and_save(cap, entity_name) # przekazujemy uchwyt do pliku wideo do funkcji detekcji i zapisu
