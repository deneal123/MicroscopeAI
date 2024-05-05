import os
import json
from library.custom_logging import setup_logging

# Назначение переменной логирования
log = setup_logging()


# Создание стандартных настроек конфигурационного файла
def save_default_config():
    """
    Создает и сохраняет стандартные настройки конфигурационного файла.

    :param: None.
    :return: None.
    """

    config_data = {

        "script_path": None,
        "temp_dir": None,

        "path_to_class_model": None,
        "path_to_detect_model": None,
        "path_to_segment_model": None,
        "path_to_detect_model_bbox": None,
        "path_to_detect_model_minibbox": None,
        "path_to_detect_model_bboxstick": None,

        "path_to_uploaded_image": None,
        "path_to_main_image": None,
        "path_to_bbox": None,
        "path_to_minibbox": None,
        "path_to_bboxstick": None,
        "path_to_inference": None,
        "path_to_font": None,

        "dir_paths": {

            "script_path": "script_path",
            "temp_dir": "temp_dir",
            "path_to_class_model": "path_to_class_model",
            "path_to_detect_model": "path_to_detect_model",
            "path_to_segment_model": "path_to_segment_model",
            "path_to_detect_model_bbox": "path_to_detect_model_bbox",
            "path_to_detect_model_minibbox": "path_to_detect_model_minibbox",
            "path_to_detect_model_bboxstick": "path_to_detect_model_bboxstick",
            "path_to_uploaded_image": "path_to_uploaded_image",
            "path_to_main_image": "path_to_main_image",
            "path_to_bbox": "path_to_bbox",
            "path_to_minibbox": "path_to_minibbox",
            "path_to_bboxstick": "path_to_bboxstick",
            "path_to_inference": "path_to_inference",
            "path_to_font": "path_to_font"
        },

        "uploaded_images": None,
        "results_detect": None,
        "results_segment": None,
        "results_detect_bbox": None,
        "results_detect_minibbox": None,
        "results_detect_bboxstick": None,
        "main_image": None,
        "bbox": None,
        "minibbox": None,
        "bboxstick": None,
        "boxes_": None,
        "boxes_bbox": None,
        "boxes_minibbox": None,
        "boxes_bboxstick": None,
        "cont_load_images": None,
        "start_time": None,
        "end_time": None,
        "execution_time": None,
        "progress": None,
        "cont_param": None,
        "radio_class_button": False,
        "apply_param_button": None,
        "clean_temp_button": None,
        "input_scale_param": None,
        "input_scale_bool": True,
        "apply_param_bool": False,
        "radio_scale_button": False,

        "dir_params": {
            "uploaded_images": "uploaded_images",
            "results_detect": "results_detect",
            "results_segment": "results_segment",
            "results_detect_bbox": "results_detect_bbox",
            "results_detect_minibbox": "results_detect_minibbox",
            "results_detect_bboxstick": "results_detect_bboxstick",
            "main_image": "main_image",
            "bbox": "bbox",
            "minibbox": "minibbox",
            "bboxstick": "bboxstick",
            "boxes_": "boxes_",
            "boxes_bbox": "boxes_bbox",
            "boxes_minibbox": "boxes_minibbox",
            "boxes_bboxstick": "boxes_bboxstick",
            "cont_load_images": "cont_load_images",
            "start_time": "start_time",
            "end_time": "end_time",
            "execution_time": "execution_time",
            "progress": "progress",
            "cont_param": "cont_param",
            "radio_class_button": "radio_class_button",
            "apply_param_button": "apply_param_button",
            "clean_temp_button": "clean_temp_button",
            "input_scale_param": "input_scale_param",
            "input_scale_bool": "input_scale_bool",
            "apply_param_bool": "apply_param_bool",
            "radio_scale_button": "radio_scale_button"
        },

        "segmented_images": [],
        "segmented_results": [],
        "list_path_to_uploaded_images": [],
        "list_path_to_bbox": [],
        "list_path_to_main_image": [],
        "list_path_to_minibbox": [],
        "list_path_to_bboxstick": [],
        "list_path_to_inference": [],
        "list_image_copy_with_mask": [],
        "scale": [],
        "stick": [],
        "name_uploaded_images": [],
        "name_uploaded_images_without_ex": [],

        "dir_lists": {
            "segmented_images": "segmented_images",
            "segmented_results": "segmented_results",
            "list_path_to_uploaded_images": "list_path_to_uploaded_images",
            "list_path_to_bbox": "list_path_to_bbox",
            "list_path_to_main_image": "list_path_to_main_image",
            "list_path_to_minibbox": "list_path_to_minibbox",
            "list_path_to_bboxstick": "list_path_to_bboxstick",
            "list_path_to_inference": "list_path_to_inference",
            "list_image_copy_with_mask": "list_image_copy_with_mask",
            "scale": "scale",
            "stick": "stick",
            "name_uploaded_images": "name_uploaded_images",
            "name_uploaded_images_without_ex": "name_uploaded_images_without_ex"
        }

    }

    with open('config.json', 'w') as config_file:
        json.dump(config_data, config_file, indent=4)

    log.info('Применение стандартных настроек конфигурационного файла')


# Глобальная переменная для хранения предыдущих параметров
previous_config_data = None


def parse_list(self, input_str):
    try:
        # Пробуем разобрать введенную строку как JSON
        list_values = json.loads(input_str)
        # Проверяем, что полученный объект - это список
        if not isinstance(list_values, list):
            raise ValueError("Введенные данные не являются списком.")
        return list_values
    except ValueError as e:
        st.error(f"Ошибка: {e}")
        return None


def load_config():
    """
    Загружает данные конфигурационного файла и сохраняет их в глобальную переменную.

    :param previous_config_file: Глобальная переменная хранящая список.
    :return config_data: Список данных выгруженных из конфигурационного файла.
    """

    global previous_config_data

    # Загрузка конфигурационных параметров из файла
    with open('config.json', 'r') as config_file:
        config_data = json.load(config_file)

    # Сохраняем текущие параметры в глобальной переменной
    previous_config_data = config_data

    return config_data


def is_value_changed(key, new_value):
    """
    Создает и сохраняет стандартные настройки конфигурационного файла.

    :param previous_config_file: Глобальная переменная хранящая список.
    :return previous_config_data[key] != new_value: Возвращает в глобальную переменную...
    Новое значение для соответствующего ключа, если это значение ключа...
    Было измененно.
    :return True: Возвращает true, если значение ключа не изменилось.
    """

    global previous_config_data

    # Проверяем, изменилось ли значение параметра
    if previous_config_data is not None and key in previous_config_data:
        return previous_config_data[key] != new_value
    else:
        # Если предыдущих данных нет, считаем, что значение изменилось
        return True


def update_config(self, **kwargs):
    """
    Обновляет конфигурационные данные на основе переданных параметров.
    """

    # Аналогичная функция _update_config, но для классов

    for key, value in kwargs.items():
        # Используем get для проверки, что ключ существует в конфигурационных данных
        if key in self.config_data and is_value_changed(key, value):
            self.config_data[key] = value

    save_config(self.config_data)


def _update_config(config_data, **kwargs):
    """
    Обновляет конфигурационные данные на основе переданных параметров.
    """

    for key, value in kwargs.items():
        # Используем get для проверки, что ключ существует в конфигурационных данных
        if key in config_data and is_value_changed(key, value):
            config_data[key] = value

    save_config(config_data)


def save_config(config_data):
    """
    Сохраняет параметры в конфигурационный файл.

    :param: None.
    :return: None.
    """

    # Сохранение конфигурационных параметров в файл
    with open('config.json', 'w') as config_file:
        json.dump(config_data, config_file, indent=4)


def default_path():
    """
    Получить путь к текущему исполняемому скрипту.

    :param: None.
    :return: None.
    """
    config_data = load_config()
    script_path = os.path.realpath(__file__)
    script_path, _ = os.path.split(script_path)
    temp_dir = os.path.join(script_path, "temp")

    paths = {
        "path_to_uploaded_image": os.path.join(temp_dir, "uploaded_image"),
        "path_to_main_image": os.path.join(temp_dir, "main_image"),
        "path_to_bbox": os.path.join(temp_dir, "bbox"),
        "path_to_minibbox": os.path.join(temp_dir, "minibbox"),
        "path_to_bboxstick": os.path.join(temp_dir, "bboxstick"),
        "path_to_inference": os.path.join(script_path, "inference"),
        "path_to_class_model": os.path.join(script_path, "weights", "weights_class", "EfficientNetB3_weights.h5"),
        "path_to_detect_model": os.path.join(script_path, "weights", "weights_detect", "last_detect_9c.pt"),
        "path_to_segment_model": os.path.join(script_path, "weights", "weights_seg", "2024_04_10_05_59_16_UEfficientNetB3_weights.h5"),
        "path_to_detect_model_bbox": os.path.join(script_path, "weights", "weights_detect", "best_detect_bbox.pt"),
        "path_to_detect_model_minibbox": os.path.join(script_path, "weights", "weights_detect", "best_detect_minibbox.pt"),
        "path_to_detect_model_bboxstick": os.path.join(script_path, "weights", "weights_detect", "best_detect_bboxstick.pt"),
        "path_to_font": os.path.join(script_path, "img", "Arial.ttf")
    }

    for key, value in paths.items():
        paths[key] = os.path.join(temp_dir, value)

    paths["script_path"] = script_path
    paths["temp_dir"] = temp_dir

    _update_config(config_data=config_data, **paths)


if __name__ == '__main__':
    save_default_config()
    default_path()
