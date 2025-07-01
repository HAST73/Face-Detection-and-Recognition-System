import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

from app_configuration import configure
import app_gui
import warnings
from torch.serialization import SourceChangeWarning

warnings.filterwarnings("ignore", category=SourceChangeWarning) # Wyłączenie ostrzeżeń związanych z Torch

# Definicja kolorów interfejsu
DARK_BG = "#2d2d2d"
MEDIUM_BG = "#3d3d3d"
BUTTON_BG = "#4d4d4d"
TEXT_COLOR = "#ffffff"
HIGHLIGHT_COLOR = "#5d5d5d"

# Główna funkcja uruchamiająca aplikację
def main():
    root = tk.Tk()
    root.title("RiPO") # Ustawienie tytułu okna
    root.geometry("400x500") # Ustawienie domyślnego rozmiaru okna
    root.configure(bg=DARK_BG) # Ustawienie koloru tła aplikacji

    # Obliczanie pozycji okna na środku ekranu
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = 400
    window_height = 500
    position_top = int(screen_height / 2 - window_height / 2)
    position_right = int(screen_width / 2 - window_width / 2)
    root.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')

    # Konfiguracja stylu dla przycisków
    style = ttk.Style()
    style.theme_use('clam')
    style.configure("TButton",
                   font=("Arial", 12),
                   padding=10,
                   background=BUTTON_BG,
                   foreground=TEXT_COLOR,
                   borderwidth=1)
    style.map('TButton',
            background=[('active', HIGHLIGHT_COLOR)]) # Zmiana koloru po najechaniu

    # Główna ramka aplikacji
    main_frame = tk.Frame(root, bg=DARK_BG)
    main_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

    # Nagłówek aplikacji
    lbl_title = tk.Label(main_frame,
                       text="RiPO",
                       font=("Arial", 20, "bold"),
                       bg=DARK_BG,
                       fg=TEXT_COLOR)
    lbl_title.pack(pady=20)

    # Dodatkowa konfiguracja stylu dla przycisków
    btn_style = ttk.Style()
    btn_style.configure('Custom.TButton',
                      font=('Arial', 12),
                      padding=15,
                      width=20,
                      background=BUTTON_BG,
                      foreground=TEXT_COLOR)

    # Przycisk do dodawania wzorca (wykrywania twarzy)
    btn_train = ttk.Button(main_frame,
                         text="Dodaj wzorzec",
                         style='Custom.TButton',
                         command=lambda: app_gui.detect_page(root)) # Przekierowanie do strony wykrywania twarzy
    btn_train.pack(pady=10)

    # Przycisk do rozpoznawania twarzy
    btn_recognize = ttk.Button(main_frame,
                             text="Rozpoznaj twarz",
                             style='Custom.TButton',
                             command=lambda: app_gui.recognize_page(root)) # Przekierowanie do strony rozpoznawania twarzy
    btn_recognize.pack(pady=10)

    # Przycisk do konfiguracji aplikacji
    btn_config = ttk.Button(main_frame,
                          text="Konfigurator",
                          style='Custom.TButton',
                          command=configure) # Otwiera konfigurator
    btn_config.pack(pady=10)

    # Uruchomienie głównej pętli aplikacji
    root.mainloop()

if __name__ == "__main__":
    main()