from flask import Flask, render_template, request, jsonify
from src.domain.models import DataObject
from src.services.recognition import RecognitionSystem

app = Flask(__name__)

# Исходная база знаний: 10 эталонных объектов для обучения алгоритмов
# Порядок признаков: [Мощность (лс), Разгон (сек), Клиренс (мм), Масса (кг), Расход (л/100км)]
training_data = [
    # Класс 1: Спорткары (высокая мощность, быстрый разгон, низкий клиренс)
    DataObject("1", [450, 3.5, 110, 1400, 15.0], 1, "Porsche 911"),
    DataObject("2", [500, 3.2, 120, 1500, 16.5], 1, "Ferrari F8"),
    DataObject("3", [380, 4.1, 130, 1350, 12.0], 1, "Toyota Supra"),
    DataObject("4", [600, 2.9, 105, 1600, 18.0], 1, "Lamborghini Huracan"),
    DataObject("5", [400, 3.9, 125, 1450, 13.5], 1, "Audi TT RS"),

    # Класс 2: Внедорожники (высокий клиренс, большая масса, медленный разгон)
    DataObject("6", [150, 11.5, 210, 2100, 9.5], 2, "Toyota RAV4"),
    DataObject("7", [200, 9.8, 220, 2300, 10.0], 2, "Mitsubishi Pajero"),
    DataObject("8", [249, 8.5, 230, 2500, 11.5], 2, "Land Cruiser 300"),
    DataObject("9", [180, 10.2, 205, 2000, 9.0], 2, "Nissan X-Trail"),
    DataObject("10", [220, 9.0, 215, 2250, 10.5], 2, "Kia Mohave")
]

# Создаем ядро распознавания и обучаем его при старте сервера
recognition_system = RecognitionSystem(training_data)


@app.route("/")
def index():
    """Отрисовка главной HTML-страницы интерфейса."""
    return render_template("index.html")


@app.route("/api/recognize", methods=["POST"])
def recognize():
    """Обработчик API, принимающий данные из веб-формы и возвращающий результат распознавания."""
    data = request.json
    try:
        # 1. Извлекаем 5 введенных пользователем признаков из JSON-запроса
        features = [
            float(data.get("f1", 0)), float(data.get("f2", 0)),
            float(data.get("f3", 0)), float(data.get("f4", 0)),
            float(data.get("f5", 0))
        ]

        # 2. Нормализуем введенные данные по заранее вычисленным максимумам
        features_norm = recognition_system.normalize_single(features)

        # 3. Запускаем оба алгоритма для принятия решения
        res_min_dist = recognition_system.predict_min_distance(features_norm)
        res_perceptron = recognition_system.predict_perceptron(features_norm)

        # 4. Отправляем ответ обратно в интерфейс
        return jsonify({
            "features_norm": [round(f, 3) for f in features_norm],
            "min_distance": res_min_dist,
            "perceptron": res_perceptron
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400