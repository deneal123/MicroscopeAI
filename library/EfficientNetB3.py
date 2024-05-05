from keras.applications import EfficientNetB3 as EfficientNetB3_
from tensorflow.keras.layers import (
    Conv2D, GlobalAveragePooling2D, Dense)
from keras.models import Model


# Модель EfficientNetB3.

def EfficientNetB3():
    # Загрузка предварительно обученной модели Xception без верхних слоев
    base_model = EfficientNetB3_(weights='imagenet', include_top=False, input_shape=(300, 300, 3))

    # Добавление своих верхних слоев для классификации 3 классов
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(4048, activation='relu')(x)
    x = Dense(4048, activation='relu')(x)
    predictions = Dense(3, activation='softmax')(x)

    # Создание новой модели с заменой верхних слоев
    model = Model(inputs=base_model.input, outputs=predictions, name='EfficientNetB3')

    return model
