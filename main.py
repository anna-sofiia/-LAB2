import os
import json
import numpy as np
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

DATASET_PATH = "data"
MODEL_PATH = "speech_model.keras"
HISTORY_PATH = "training_history.json"
FEATURES_PATH = "dataset_features.npz"

SAMPLE_RATE = 16000
MAX_PAD_LEN = 32
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 25
BATCH_SIZE = 16

# Класи: слова цифр + звичайні слова
CLASSES = [
    "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine",
    "right", "house", "bird"
]


# ВИТЯГНЕННЯ MFCC-ОЗНАК З ОДНОГО АУДІОФАЙЛУ
def extract_mfcc(file_path, max_pad_len=MAX_PAD_LEN):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]

        return mfcc

    except Exception as e:
        print(f"Помилка при обробці файлу {file_path}: {e}")
        return None


# ПОВНЕ ЗАВАНТАЖЕННЯ ДАТАСЕТУ З ПАПОК
def load_dataset(dataset_path):
    X = []
    y = []

    for label in CLASSES:
        folder_path = os.path.join(dataset_path, label)

        if not os.path.exists(folder_path):
            print(f"Увага: папка {folder_path} не знайдена")
            continue

        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(".wav"):
                file_path = os.path.join(folder_path, file_name)
                features = extract_mfcc(file_path)

                if features is not None:
                    X.append(features)
                    y.append(label)

    X = np.array(X)
    y = np.array(y)

    return X, y


# ЗАВАНТАЖИТИ ОБРОБЛЕНИЙ ДАТАСЕТ АБО СТВОРИТИ ЙОГО
def load_or_create_dataset(dataset_path, features_path=FEATURES_PATH):
    if os.path.exists(features_path):
        print("Завантаження вже обробленого датасету...")
        data = np.load(features_path, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        return X, y

    X, y = load_dataset(dataset_path)

    if len(X) == 0:
        return X, y

    np.savez(features_path, X=X, y=y)
    print(f"Оброблений датасет збережено у файл: {features_path}")

    return X, y


# ПЕРЕГЕНЕРАЦІЯ КЕШУ ДАТАСЕТУ
def rebuild_dataset_cache():
    if os.path.exists(FEATURES_PATH):
        os.remove(FEATURES_PATH)
        print("Старий кеш датасету видалено.")

    X, y = load_or_create_dataset(DATASET_PATH)

    if len(X) == 0:
        print("Не вдалося створити кеш датасету.")
    else:
        print("Кеш датасету успішно створено.")


# ПІДГОТОВКА ДАНИХ ДЛЯ CNN
def prepare_data(X, y):
    y_encoded = np.array([CLASSES.index(label) for label in y])
    y_categorical = to_categorical(y_encoded, num_classes=len(CLASSES))

    X = np.expand_dims(X, axis=-1)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_categorical,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )

    return X_train, X_test, y_train, y_test


# ПОБУДОВА CNN-МОДЕЛІ
def build_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ЗБЕРЕЖЕННЯ ІСТОРІЇ НАВЧАННЯ
def save_history(history, history_path=HISTORY_PATH):
    try:
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history.history, f, ensure_ascii=False, indent=4)
        print(f"Історію навчання збережено у файл: {history_path}")
    except Exception as e:
        print(f"Помилка при збереженні історії: {e}")


# ЗЧИТУВАННЯ ІСТОРІЇ НАВЧАННЯ
def load_history(history_path=HISTORY_PATH):
    if not os.path.exists(history_path):
        print("Файл історії навчання не знайдено.")
        return None

    try:
        with open(history_path, "r", encoding="utf-8") as f:
            history_data = json.load(f)
        return history_data
    except Exception as e:
        print(f"Помилка при зчитуванні історії: {e}")
        return None


# ПОБУДОВА ГРАФІКІВ ЗА ЗБЕРЕЖЕНОЮ ІСТОРІЄЮ
def plot_history_from_dict(history_dict):
    if history_dict is None:
        print("Немає даних для побудови графіків.")
        return

    if "accuracy" in history_dict and "val_accuracy" in history_dict:
        plt.figure(figsize=(8, 5))
        plt.plot(history_dict["accuracy"], label="Точність на train")
        plt.plot(history_dict["val_accuracy"], label="Точність на validation")
        plt.xlabel("Епоха")
        plt.ylabel("Accuracy")
        plt.title("Графік точності")
        plt.legend()
        plt.grid()
        plt.show()
    else:
        print("У файлі історії немає accuracy / val_accuracy.")

    if "loss" in history_dict and "val_loss" in history_dict:
        plt.figure(figsize=(8, 5))
        plt.plot(history_dict["loss"], label="Втрати на train")
        plt.plot(history_dict["val_loss"], label="Втрати на validation")
        plt.xlabel("Епоха")
        plt.ylabel("Loss")
        plt.title("Графік втрат")
        plt.legend()
        plt.grid()
        plt.show()
    else:
        print("У файлі історії немає loss / val_loss.")


# ВИБІР АУДІОФАЙЛУ ЧЕРЕЗ ВІКНО
def choose_audio_file():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_path = filedialog.askopenfilename(
        title="Оберіть аудіофайл",
        filetypes=[("WAV files", "*.wav")]
    )

    root.destroy()
    return file_path


# НАВЧАННЯ НОВОЇ МОДЕЛІ
def train_new_model():
    print("Завантаження датасету...")
    X, y = load_or_create_dataset(DATASET_PATH)

    if len(X) == 0:
        print("Датасет порожній або .wav файли не знайдені.")
        return None

    print(f"Кількість зразків: {len(X)}")
    print(f"Форма X: {X.shape}")
    print(f"Перші мітки: {y[:10]}")

    X_train, X_test, y_train, y_test = prepare_data(X, y)

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    model = build_model(input_shape=X_train.shape[1:], num_classes=len(CLASSES))

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    model.save(MODEL_PATH)
    print(f"Модель збережена у файл: {MODEL_PATH}")

    save_history(history)
    plot_history_from_dict(history.history)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nТочність на тестових даних: {accuracy:.4f}")
    print(f"Втрати на тестових даних: {loss:.4f}")

    return model


# ЗАВАНТАЖЕННЯ ВЖЕ ГОТОВОЇ МОДЕЛІ
def load_saved_model():
    if not os.path.exists(MODEL_PATH):
        print("Збережена модель не знайдена.")
        return None

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Модель успішно завантажено.")
        return model
    except Exception as e:
        print(f"Помилка при завантаженні моделі: {e}")
        return None


# ОЦІНКА МОДЕЛІ НА ТЕСТОВИХ ДАНИХ
def evaluate_model(model):
    if model is None:
        print("Спочатку потрібно завантажити або навчити модель.")
        return

    print("Завантаження датасету для оцінки...")
    X, y = load_or_create_dataset(DATASET_PATH)

    if len(X) == 0:
        print("Датасет порожній або .wav файли не знайдені.")
        return

    _, X_test, _, y_test = prepare_data(X, y)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nТочність на тестових даних: {accuracy:.4f}")
    print(f"Втрати на тестових даних: {loss:.4f}")


# ПЕРЕДБАЧЕННЯ ДЛЯ ОДНОГО WAV-ФАЙЛУ
def predict_audio(model, file_path):
    if model is None:
        print("Спочатку потрібно завантажити або навчити модель.")
        return None

    features = extract_mfcc(file_path)

    if features is None:
        return None

    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=-1)

    prediction = model.predict(features, verbose=0)
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_label = CLASSES[predicted_index]

    return predicted_label


# МЕНЮ
def print_menu():
    print("\nМЕНЮ ПРОГРАМИ")
    print("1 - Навчити нову модель")
    print("2 - Завантажити збережену модель")
    print("3 - Оцінити модель на тестових даних")
    print("4 - Розпізнати новий .wav файл")
    print("5 - Показати графіки останнього навчання")
    print("6 - Перегенерувати кеш датасету")
    print("0 - Вийти")


# ГОЛОВНА ФУНКЦІЯ
def main():
    model = None

    while True:
        print_menu()
        choice = input("Оберіть пункт меню: ").strip()

        if choice == "1":
            model = train_new_model()

        elif choice == "2":
            model = load_saved_model()

        elif choice == "3":
            evaluate_model(model)

        elif choice == "4":
            if model is None:
                print("Модель ще не завантажена. Спробую завантажити збережену модель...")
                model = load_saved_model()

            if model is None:
                continue

            test_file = choose_audio_file()

            if not test_file:
                print("Файл не обрано.")
                continue

            result = predict_audio(model, test_file)

            if result is not None:
                print(f"Передбачення моделі: {result}")

        elif choice == "5":
            history_data = load_history()
            plot_history_from_dict(history_data)

        elif choice == "6":
            rebuild_dataset_cache()

        elif choice == "0":
            print("Завершення програми.")
            break

        else:
            print("Невірний пункт меню. Спробуй ще раз.")


if __name__ == "__main__":
    main()