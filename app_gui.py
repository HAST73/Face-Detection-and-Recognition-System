import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

from app_recognition import recognize_camera, recognize_video
from app_detection import detect_camera, detect_video
from app_configuration import configure

DARK_BG = "#2d2d2d"
MEDIUM_BG = "#3d3d3d"
BUTTON_BG = "#4d4d4d"
TEXT_COLOR = "#ffffff"
HIGHLIGHT_COLOR = "#5d5d5d"


# Konfiguracja stylu dla przycisków i pól tekstowych
def setup_style():
    style = ttk.Style()
    style.theme_use('clam')
    # Styl dla przycisków
    style.configure("TButton",
                    font=("Arial", 12),
                    padding=10,
                    background=BUTTON_BG,
                    foreground=TEXT_COLOR,
                    borderwidth=1)
    style.map('TButton',
              background=[('active', HIGHLIGHT_COLOR)]) # Zmiana koloru przycisku po najechaniu

    # Styl dla pól tekstowych
    style.configure("TEntry",
                    fieldbackground=MEDIUM_BG,
                    foreground=TEXT_COLOR,
                    borderwidth=2)


# Przywraca główne menu aplikacji
def return_to_main(frame, root):
    for widget in frame.winfo_children():
        widget.destroy() # Usuwa wszystkie widżety w bieżącej ramce

    frame.configure(bg=DARK_BG) # Ustawienie koloru tła

    # Nagłówek aplikacji
    lbl_title = tk.Label(frame,
                         text="RiPO",
                         font=("Arial", 20, "bold"),
                         bg=DARK_BG,
                         fg=TEXT_COLOR)
    lbl_title.pack(pady=20)

    # Definicja przycisków menu głównego
    buttons = [
        ("Dodaj wzorzec", lambda: detect_page(root)),
        ("Rozpoznaj twarz", lambda: recognize_page(root)),
        ("Konfigurator", configure)
    ]

    for text, cmd in buttons:
        btn = ttk.Button(frame,
                         text=text,
                         style='Custom.TButton',
                         command=cmd)
        btn.pack(pady=10)


# Strona dodawania wzorca (wykrywanie twarzy)
def detect_page(root):
    setup_style() # Ustawienie stylu interfejsu
    for widget in root.winfo_children():
        widget.destroy() # Usunięcie poprzedniego widoku

    # Tworzenie głównej ramki
    main_frame = tk.Frame(root, bg=DARK_BG)
    main_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

    # Nagłówek strony
    lbl_title = tk.Label(main_frame,
                         text="Dodawanie wzorca",
                         font=("Arial", 16),
                         bg=DARK_BG,
                         fg=TEXT_COLOR)
    lbl_title.pack(pady=15)

    # Sekcja wpisywania nazwy wzorca
    entry_frame = tk.Frame(main_frame, bg=DARK_BG)
    entry_frame.pack(pady=10)

    lbl_pattern = tk.Label(entry_frame,
                           text="Nazwa wzorca:",
                           font=("Arial", 12),
                           bg=DARK_BG,
                           fg=TEXT_COLOR)
    lbl_pattern.pack(side=tk.LEFT)

    entry_text = tk.StringVar()
    entry_field = ttk.Entry(entry_frame,
                            textvariable=entry_text,
                            style='TEntry')
    entry_field.pack(side=tk.LEFT, padx=10) # Pole tekstowe do wpisania nazwy wzorca

    # Ramka dla przycisków
    btn_frame = tk.Frame(main_frame, bg=DARK_BG)
    btn_frame.pack(pady=20)

    # Przycisk wyboru źródła dla detekcji
    sources = [
        ("Kamera", lambda: detect_camera(entry_text.get())), # Wykrywanie twarzy przez kamerę
        ("Wideo", lambda: detect_video(entry_text.get())), # Wykrywanie twarzy na wideo
        ("Powrót", lambda: return_to_main(main_frame, root)) # Powrót do menu głównego
    ]

    for text, cmd in sources:
        btn = ttk.Button(btn_frame,
                         text=text,
                         style='Custom.TButton',
                         command=cmd)
        btn.pack(pady=8, fill=tk.X) # Dodanie przycisków do wyboru źródła


# Strona rozpoznawania twarzy
def recognize_page(root):
    setup_style() # Ustawienie stylu interfejsu
    for widget in root.winfo_children():
        widget.destroy() # Usunięcie poprzedniego widoku

    # Tworzenie głównej ramki
    main_frame = tk.Frame(root, bg=DARK_BG)
    main_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

    # Nagłówek strony
    lbl_title = tk.Label(main_frame,
                         text="Rozpoznawanie twarzy",
                         font=("Arial", 16),
                         bg=DARK_BG,
                         fg=TEXT_COLOR)
    lbl_title.pack(pady=15)

    # Ramka dla przycisków
    btn_frame = tk.Frame(main_frame, bg=DARK_BG)
    btn_frame.pack(pady=20)

    # Przycisk wyboru źródła dla rozpoznawania
    sources = [
        ("Kamera", recognize_camera), # Rozpoznawanie twarzy przez kamerę
        ("Wideo", recognize_video), # Rozpoznawanie twarzy na wideo
        ("Powrót", lambda: return_to_main(main_frame, root)) # Powrót do menu głównego
    ]

    for text, cmd in sources:
        btn = ttk.Button(btn_frame,
                         text=text,
                         style='Custom.TButton',
                         command=cmd)
        btn.pack(pady=8, fill=tk.X) # Dodanie przycisków