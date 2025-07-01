import os  # obsługa systemu plików
import sys  # obsługa funkcji systemowych
from tkinter import filedialog  # okno do wyboru plików wideo

import logging # conf do logga

mpl_logger = logging.getLogger('matplotlib')  # wyłączamy logowanie dla matplotlib
mpl_logger.setLevel(logging.WARNING)
import logging.config

logging.config.fileConfig("config/logging.conf") # konfiguracja logowania z pliku konfiguracyjnego logging.conf
logger = logging.getLogger('api')  # Ustawiamy logger o nazwie 'api'

import yaml  # Obsługa plików YAML
import cv2 # biblioteka do przetwarzania obrazu i wideo
import numpy as np

# importujemy klasy do ładowania i obsługi modeli detekcji, wyrównania i rozpoznawania twarzy
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper
from core.model_loader.face_recognition.FaceRecModelLoader import FaceRecModelLoader
from core.model_handler.face_recognition.FaceRecModelHandler import FaceRecModelHandler
from collections import defaultdict # słownik do grupowania cech

with open('config/model_conf.yaml') as f:
    model_conf = yaml.load(f, Loader=yaml.SafeLoader)  # załadowanie konfiguracji modeli

def load_models():
    model_path = 'models'  # ścieżka do folderu z modelami
    scene = 'non-mask'  # scena bez maski
    try:
        # model detekcji twarzy
        face_det_loader = FaceDetModelLoader(model_path, 'face_detection', model_conf[scene]['face_detection'])
        face_det_model, face_det_cfg = face_det_loader.load_model()
        face_det_handler = FaceDetModelHandler(face_det_model, 'cuda:0', face_det_cfg)

        # model wyrównania twarzy
        face_align_loader = FaceAlignModelLoader(model_path, 'face_alignment', model_conf[scene]['face_alignment'])
        face_align_model, face_align_cfg = face_align_loader.load_model()
        face_align_handler = FaceAlignModelHandler(face_align_model, 'cuda:0', face_align_cfg)

        # model rozpoznawania twarzy
        face_rec_loader = FaceRecModelLoader(model_path, 'face_recognition', model_conf[scene]['face_recognition'])
        face_rec_model, face_rec_cfg = face_rec_loader.load_model()
        face_rec_handler = FaceRecModelHandler(face_rec_model, 'cuda:0', face_rec_cfg)

    except Exception as e:
        print(f'Error loading models: {e}')  # błąd wczytywania modeli
        sys.exit(-1)  # zakończenie programu

    return face_det_handler, face_align_handler, face_rec_handler


def extract_face_feature(image, face_det_handler, face_align_handler, face_rec_handler):

    dets = face_det_handler.inference_on_image(image)  # detekcja twarzy na obrazie
    if len(dets) == 0:  # jeśli nie wykryto twarzy, zwracamy None
        return None

    landmarks = face_align_handler.inference_on_image(image, dets[0])  # wyrównanie twarzy - landmarki
    face_cropper = FaceRecImageCropper()  # inicjalizacja croppera twarzy
    cropped_face = face_cropper.crop_image_by_mat(image, landmarks.flatten().tolist())

    return face_rec_handler.inference_on_image(cropped_face)  # wyciąganie cech twarzy



def recognize(cap):
    face_det_handler, face_align_handler, face_rec_handler = load_models() # załadowane modele do detekcji, wyrównywania i rozpoznawania

    path_to_imgs = './collected_images' # ścieżka do obrazków
    stored_images = [os.path.join(path_to_imgs, f) for f in os.listdir(path_to_imgs) if f.endswith('.jpg')] # zebranie obrazków jpeg do tablicy

    features_dict = defaultdict(list) # stworzenie słownika: {nazwa_osoby: [lista wektorów cech ze zdjęć]}

    for img_path in stored_images: # iteracja po każdym zdjęciu w folderze
        image = cv2.imread(img_path) # wczytanie konkretnego zdjęcia
        person_name = os.path.basename(img_path).split("_")[0]  # nazwa osoby z pliku
        feature = extract_face_feature(image, face_det_handler, face_align_handler, face_rec_handler) # pobieranie unikalnych cech twarzy ze zdjęcia
        if feature is not None:
            features_dict[person_name].append(feature) # zapis cech pod nazwą osoby

    if not features_dict: # jeżeli nie ma żadnych cech
        logger.warning("Brak zapisanych wzorców twarzy.") # koniec programu
        return

    stored_features = [] # lista do przechowywania finalnych cech
    stored_names = [] # lista do przechowywania nazw osób

    def l2_normalization(x):
        return x / np.linalg.norm(x) # normalizacja cech- dzielenie wektora x przez jego długość

    for name, features in features_dict.items(): # iteracja po osobach
        mean_feature = np.mean(features, axis=0) # obliczanie średniego wektora cech osoby
        stored_features.append(l2_normalization(mean_feature)) # do listy wzorców dodany jest znormalizowany średni wektor
        stored_names.append(name) # dodanie nazwy osoby do listy

    stored_matrix = np.stack(stored_features) # zamiana listy na macierz do szybszego porównywania
    Min_score = 0.5 # minimalny próg rozpoznania

    frame_count = 0 # przetworzone klatki
    last_box, last_name, last_score = None, None, None # dane ostatniego dopasowania

    while True:
        ret, frame = cap.read() # pobranie klatki z kamery
        if not ret:
            break

        frame_count += 1
        process_this = frame_count % 3 == 0 # przetwarzanie co trzecią klatkę dla wydajności

        if process_this:
            try:
                detections = face_det_handler.inference_on_image(frame) # detekcja twarzy
            except Exception as e:
                logger.error('Face detection failed:', exc_info=True) # błąd w detekcji
                break

            if len(detections) > 0:
                box = list(map(int, detections[0])) # współrzędne ramki z detekcji
                characteristic_features = face_align_handler.inference_on_image(frame, detections[0]) # cechy charakterystyczne- nos, oczy
                face_cropper = FaceRecImageCropper() # wycinacz twarzy
                cropped_face = face_cropper.crop_image_by_mat(frame, characteristic_features.flatten().tolist()) # wycięcie twarzy z obrazka, aby uniknąć tła przy rozpoznawaniu
                live_feature = face_rec_handler.inference_on_image(cropped_face) # tworzenie z wyciętej twarzy wektor
                live_feature = l2_normalization(live_feature) # normalizacja wektora do porównywania z tym ze wzoru

                scores = np.dot(stored_matrix, live_feature) # obliczanie podobieństwa

                best_idx = np.argmax(scores) # index najlepszego dopasowania
                best_score = scores[best_idx] # najlepszy wynik
                best_name = stored_names[best_idx] # osoba najlepiej pasująca do aktualnego obrazu twarzy

                if best_score > Min_score: # jeżeli wynik przekroczy próg
                    last_name = best_name # podmiana nazwy
                    last_score = best_score # podmiana wyniku
                else:
                    last_name = "Nie rozpoznano" # jeżeli jest niższy to nie rozpoznano twarzy
                    last_score = best_score # podmiana wyniku

                last_box = box # ramka twarzy
            else:
                last_box, last_name, last_score = None, None, None # przypadek, gdy nie ma detekcji

        if last_box: # jeżeli twarz jest wykryta
            cv2.rectangle(frame, (last_box[0], last_box[1]), (last_box[2], last_box[3]), (0, 255, 0), 2) # rysowanie ramki zielonej na twarzy
        if last_name: # jeżeli wynik jest wiadomy
            cv2.putText(frame, f'{last_name} ({last_score:.2f})', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2) # wyświetlenie wyniku w oknie

        # obliczanie jasności sceny
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # obraz z samym światłem bez kolorów
        brightness = np.mean(gray) # uśrednienie wartości, aby uzyskać zakres [0-czarny,255-biały]

        # Logowanie rozpoznanie i poziom jasności do logów
        if last_name:
            logger.info(f"Oświetlenie: {brightness:.2f}, Rozpoznano: {last_name}, Score: {last_score:.3f}")
        else:
            logger.info(f"Oświetlenie: {brightness:.2f}, Twarz nie została rozpoznana.")

        # wyświetlenie klatki
        cv2.imshow('Rozpoznawanie Twarzy', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): # wyjście za pomocą q
            break

    cap.release() # zamkniecie połączenia z kamerą
    cv2.destroyAllWindows() # zamknięcie okna GUI


def recognize_camera():
    cap = cv2.VideoCapture(0) # domyślna kamera
    if not cap.isOpened(): # sprawdzamy, czy kamera została poprawnie otwarta
        logger.error("Failed to open camera.") # komunikacja o błędzie
        return

    recognize(cap) # przekazanie do funkcji rozpoznawania twarzy


def recognize_video():
    video_path = filedialog.askopenfilename(title="Wybierz plik wideo.", filetypes=[("Video Files", "*.mp4;*.avi;*.mkv")]) # wypisane dane typy plików obsługiwanych

    if not video_path: # czy wybraliśmy plik
        logger.warning("No video file selected.") # komunikat o błędzie
        return

    cap = cv2.VideoCapture(video_path) # połączenie z plikiem video

    if not cap.isOpened(): # czy źle zostało otwarte połączenie z video
        logger.error("Failed to open video file.") # komunikat o błędzie
        return

    recognize(cap) # przekazanie do funkcji rozpoznawania twarzy
