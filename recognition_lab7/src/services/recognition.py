from typing import List, Tuple
from src.domain.models import DataObject


class RecognitionSystem:
    """
        Главный класс бизнес-логики (математическое ядро системы).
    """
    def __init__(self, training_data: List[DataObject]):
        """Инициализация системы распознавания при запуске приложения."""

        # Сохраняем исходное обучающее множество (наши 10 автомобилей)
        self.training_data = training_data

        # Автоматически определяем количество признаков 5,
        # посмотрев на длину массива признаков у самого первого объекта из выборки
        self.feature_count = len(training_data[0].features)

        # 1. Поиск максимумов для нормализации данных
        self.max_values = self._find_max_values()

        # 2. Нормализация обучающей выборки (приведение к диапазону 0..1)
        self.norm_data = self._normalize_dataset(self.training_data)

        # 3. Вычисление эталонов (центров) классов для метода мин. расстояния
        self.prototypes = self._calculate_prototypes()

        # 4. Инициализация и обучение алгоритма восприятия (перцептрона)
        self.weights = [0.0] * self.feature_count
        self.w0 = 0.0
        self._train_perceptron()

    def _find_max_values(self) -> List[float]:
        """Ищет максимальное значение по каждому из признаков среди всех объектов."""
        max_vals = [0.0] * self.feature_count
        for obj in self.training_data:
            for i in range(self.feature_count):
                if obj.features[i] > max_vals[i]:
                    max_vals[i] = obj.features[i]
        return max_vals

    def _normalize_dataset(self, dataset: List[DataObject]) -> List[DataObject]:
        """Делит каждый признак на его максимум, чтобы все данные были в одном масштабе."""
        norm_dataset = []
        for obj in dataset:
            norm_features = [
                obj.features[i] / self.max_values[i] if self.max_values[i] != 0 else 0
                for i in range(self.feature_count)
            ]
            norm_dataset.append(DataObject(obj.id, norm_features, obj.class_label, obj.name))
        return norm_dataset

    def normalize_single(self, features: List[float]) -> List[float]:
        """Подготавливает данные нового (введенного пользователем) объекта для алгоритмов."""
        return [
            features[i] / self.max_values[i] if self.max_values[i] != 0 else 0
            for i in range(self.feature_count)
        ]

    def _calculate_prototypes(self) -> dict:
        """Считает среднее арифметическое всех признаков для каждого класса (создает "идеального" представителя)."""
        prototypes = {1: [0.0] * self.feature_count, 2: [0.0] * self.feature_count}
        counts = {1: 0, 2: 0}

        # Суммируем признаки по классам
        for obj in self.norm_data:
            c = obj.class_label
            counts[c] += 1
            for i in range(self.feature_count):
                prototypes[c][i] += obj.features[i]

        # Делим на количество объектов в классе
        for c in prototypes:
            if counts[c] > 0:
                prototypes[c] = [val / counts[c] for val in prototypes[c]]
        return prototypes

    def predict_min_distance(self, features_norm: List[float]) -> dict:
        """Определяет класс на основе минимального расстояния до прототипов."""
        d_values = {}
        for c, p in self.prototypes.items():
            # Математическая формула решающей функции Di = 2*SUM(X*P) - SUM(P*P)
            sum_xp = sum(features_norm[i] * p[i] for i in range(self.feature_count))
            sum_pp = sum(p[i] * p[i] for i in range(self.feature_count))
            d_values[c] = 2 * sum_xp - sum_pp

        # Побеждает класс с наибольшим значением функции D
        predicted_class = 1 if d_values[1] > d_values[2] else 2
        return {
            "predicted_class": predicted_class,
            "d1": round(d_values[1], 3),
            "d2": round(d_values[2], 3)
        }

    def _train_perceptron(self):
        """Обучает разделяющую границу (гиперплоскость) между двумя классами."""
        self.weights = [0.0] * self.feature_count  # Веса признаков
        self.w0 = 0.0  # Свободный член

        n_objects = len(self.norm_data)
        e_counter = 0  # Счетчик объектов, классифицированных без ошибок
        max_iterations = 10000
        iterations = 0

        # Обучаем, пока алгоритм не перестанет ошибаться на тестовой выборке
        while e_counter < n_objects and iterations < max_iterations:
            for obj in self.norm_data:
                # Вычисление текущего значения решающей функции D12
                d12 = sum(self.weights[i] * obj.features[i] for i in range(self.feature_count)) + self.w0

                # Проверка: правильно ли угадан класс
                if (obj.class_label == 1 and d12 > 0) or (obj.class_label == 2 and d12 < 0):
                    e_counter += 1
                else:
                    # Корректировка весов, если алгоритм ошибся на 1-м классе
                    if obj.class_label == 1 and d12 <= 0:
                        for i in range(self.feature_count):
                            self.weights[i] += obj.features[i]
                        self.w0 += 1.0
                    # Корректировка весов, если алгоритм ошибся на 2-м классе
                    elif obj.class_label == 2 and d12 >= 0:
                        for i in range(self.feature_count):
                            self.weights[i] -= obj.features[i]
                        self.w0 -= 1.0
                    e_counter = 0  # Сбрасываем счетчик при любой ошибке
            iterations += 1

    def predict_perceptron(self, features_norm: List[float]) -> dict:
        """Классифицирует новый объект по уже обученному уравнению."""
        d12 = sum(self.weights[i] * features_norm[i] for i in range(self.feature_count)) + self.w0
        # Если значение D12 больше 0 — это Класс 1, если меньше — Класс 2
        predicted_class = 1 if d12 > 0 else 2
        return {
            "predicted_class": predicted_class,
            "d12": round(d12, 3),
            "equation": f"D12 = {' + '.join([f'{round(w, 2)}*X{i + 1}' for i, w in enumerate(self.weights)])} + {round(self.w0, 2)}"
        }