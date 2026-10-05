from dataclasses import dataclass
from typing import List

@dataclass
class DataObject:
    """Модель распознаваемого объекта."""
    id: str
    features: List[float] # Массив числовых признаков (мощность, разгон и т.д.)
    class_label: int      # Метка класса: 1 (Спорткар) или 2 (Внедорожник)
    name: str = ""        # Текстовое название для удобства