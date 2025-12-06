# Skanownik - Vehicle Undercarriage Anomaly Detection

System wykrywania anomalii w podwoziach pojazdów (skany RTG) z podziałem na samochody osobowe i dostawcze. Projekt wykorzystuje PyTorch Lightning oraz podejście oparte na ekstrakcji cech (ResNet18) i algorytmie KNN (Memory Bank) do wykrywania odchyleń od normy.

## 📂 Struktura Projektu

```text
backend/
├── data/                       # Dane treningowe i testowe
│   ├── czyste/                 # Obrazy referencyjne (bez anomalii)
│   └── brudne/                 # Obrazy z anomaliami (do testów)
├── src/
│   ├── dataloader/             # Moduły ładowania danych (DataModules)
│   ├── models/                 # Definicje modeli (Classifier, AnomalyDetector)
│   ├── utils/                  # Narzędzia pomocnicze (wizualizacja)
│   ├── inference_pipeline.py   # Główny skrypt inferencji
│   ├── train.py                # Trening klasyfikatora typu pojazdu
│   ├── train_anomaly_*.py      # Trening detektorów anomalii
├── trained_models/             # Zapisane wagi klasyfikatora
├── trained_models_anomaly/     # Zapisane wagi detektorów i configi progów
├── run_inference_test.py       # Skrypt testowy (losowe próbki)
└── pyproject.toml              # Konfiguracja zależności (uv)
```

## 🚀 Instalacja i Konfiguracja

Projekt używa **uv** do zarządzania zależnościami.

1. **Zainstaluj uv** (jeśli nie masz):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Zainstaluj zależności**:
   ```bash
   uv sync
   ```

## 🏃‍♂️ Uruchamianie

### 1. Trening (Opcjonalnie)
Modele są już wytrenowane, ale jeśli chcesz przetrenować je na nowych danych:

*   **Klasyfikator Typu Pojazdu** (Osobowe vs Dostawcze):
    ```bash
    uv run src/train.py
    ```
*   **Detektor Anomalii (Osobowe)**:
    ```bash
    uv run src/train_anomaly_osobowe.py
    ```
*   **Detektor Anomalii (Dostawcze)**:
    ```bash
    uv run src/train_anomaly_dostawcze.py
    ```
    *Po treningu pliki konfiguracyjne z progami (`config_*.json`) zostaną automatycznie zaktualizowane w `trained_models_anomaly/`.*

### 2. Inferencja (Pojedynczy plik)
Uruchomienie pełnego potoku (klasyfikacja -> detekcja -> wizualizacja) dla jednego zdjęcia:

```bash
uv run src/inference_pipeline.py data/brudne/osobowe/przykladowy_skan.png
```
Wynik zostanie wyświetlony w oknie z zaznaczonym obszarem anomalii.

### 3. Testy Automatyczne
Uruchomienie testu na losowej próbce zdjęć (np. tylko "brudnych") w celu weryfikacji skuteczności:

```bash
uv run run_inference_test.py
```

## ⚙️ Konfiguracja Progów
Progi detekcji (Thresholds) są ładowane dynamicznie z plików JSON generowanych podczas treningu:
*   `trained_models_anomaly/config_osobowe.json`
*   `trained_models_anomaly/config_dostawcze.json`

Możesz ręcznie edytować te pliki, aby zmienić czułość detekcji (`threshold` vs `safe_threshold`).
