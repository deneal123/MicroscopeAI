import streamlit as st
from library.custom_logging import setup_logging
from library.IndentationHelper import IndentationHelper
from config_file import load_config, is_value_changed, save_config
from PIL import Image, ImageDraw, ImageFont
import string
from ultralytics import YOLO
import os
import torchvision.transforms as transforms
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import math
import cv2
from library.components import add_text_to_image
from datetime import datetime
import time
import easyocr
import shutil
from library.BoxPlot import BOXPLOT
from stqdm import stqdm
import library.style_button as sb
from library.tf_keras_predict import predict
from library.vgg_unet import vgg_unet
from tensorflow.keras.models import load_model
import tensorflow as tf
import json


class PageTools:

    def __init__(self):

        # Назначение переменной логирования
        self.log = setup_logging()
        # Загрузка конфигурационного файла
        self.config_data = load_config()
        # Класс функций для отступов
        self.helper = IndentationHelper()
        # Определяем время инициализации класса
        self._time()
        # Обновляем изменяемые переменные после предыдущего запуска
        self._init_params()
        # Выгружаем и инициализируем переменные стандартных путей
        self._init_default_path()
        # Создаем директории, если они не существуют
        self._make_os_dir()
        # Инициализируем листы
        self._init_list()
        # Инициализируем словари
        self._init_dir()

        # Получаем путь к домашней папке пользователя
        home_path = os.path.expanduser("~")
        path_to_easyocr_user_network = os.path.join(home_path, ".EasyOCR", "user_network")
        path_to_easyocr_model = os.path.join(home_path, ".EasyOCR", "model")

        self.reader = easyocr.Reader(['en'], recog_network='minibbox_ocr',
                                     user_network_directory=path_to_easyocr_user_network,
                                     model_storage_directory=path_to_easyocr_model)

        self.detect_model_bbox = YOLO(f"{self.path_to_detect_model_bbox}")
        self.detect_model_bboxstick = YOLO(f"{self.path_to_detect_model_bboxstick}")
        self.detect_model_minibbox = YOLO(f"{self.path_to_detect_model_minibbox}")
        self.detect_model = YOLO(f"{self.path_to_detect_model}")

        self.segment_model = vgg_unet(n_classes=2, input_height=224, input_width=224)
        self.segment_model.load_weights(self.path_to_segment_model)

    def _time(self):
        self.timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")

    def _init_default_path(self):
        paths = self.config_data["dir_paths"]
        for key, value in paths.items():
            setattr(self, key, self.config_data[f"{value}"])

    def _init_params(self):
        params = self.config_data["dir_params"]
        for key, value in params.items():
            setattr(self, key, self.config_data[f"{value}"])

    def _init_list(self):
        lists = self.config_data["dir_lists"]
        for key, value in lists.items():
            setattr(self, key, self.config_data[f"{value}"])

    def _init_dir(self):
        # Словарь для хранения информации о вычислениях
        self.dir_data = {}
        # Словарь для хранения количества сфер для каждого изображения
        self.dir_num_sphere_for_every_image = {}
        # Словарь для хранения значений диаметров для каждого изображения
        self.dir_diameters_for_every_image = {}
        # Словарь для хранения значений масштабов для каждого изображения
        self.dir_scales_for_every_image = {}

    def _refresh_inference_params(self):
        self.boxes_ = []
        self.segmented_images = []
        self.segmented_results = []

    def save_to_json(self, data, file_path):
        """
        Сохраняет данные в формате JSON в указанный файл.

        Параметры:
        - data: словарь или список данных для сохранения в JSON.
        - file_path: строка, путь к файлу JSON.

        Возвращает:
        - None
        """

        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

    def add_key_value(self, dictionary, key, value):
        """
        Добавляет новый ключ и значение в словарь.

        Параметры:
        - dictionary: словарь, в который добавляется новый ключ и значение.
        - key: ключ для добавления.
        - value: значение для добавления.

        Возвращает:
        - dictionary: словарь с добавленным ключом и значением.
        """
        dictionary[key] = value
        return dictionary

    def run(self):
        """
        Запуск приложения.
        """

        # Содержимое страницы
        self.title_page()

        # Контейнер ввода параметров
        self.input_param_container()

        # Контейнер для загрузки изображений
        self._load_images()

        # Подготовка данных
        if self.apply_param_button:
            self._prepare_data()

            # Построение боксплота
            self._boxplot_num_sphere_all_image()

            # Подсчет времени работы алгоритма
            self._calculate_time_work()

    def title_page(self):
        """
        Содержимое страницы ввиде вступительного текста.
        """
        self.helper.create_indentations(1)
        self.progress = st.container()

    def _warning_input_scale_param(self):
        if self.input_scale_param != '':
            elements = self.input_scale_param.split()
            try:
                # Пробуем преобразовать каждый элемент в целое число
                self.scale = [int(element) for element in elements]
                self.input_scale_bool = False
            except ValueError:
                # Если возникает ошибка ValueError (невозможно преобразовать в int),
                # выводим предупреждение пользователю
                self.progress.warning("Будет применено автоматическое распознавание масштаба.")
                self.progress.info("(Масштаб в формате a, b,..., где"
                                   "a,b,... - значение нанометров в одном пикселе для каждого изображения)")

    def input_param_container(self):

        # Контейнер для ввода параметров
        self.cont_param = st.container()

        # Создание двух столбцов в контейнере
        column_1, column_2, column_3, column_4 = self.cont_param.columns(4)

        with column_1:
            # Кнопка для запуска вычислений
            sb._center()
            self.apply_param_button = st.button("Запустить вычисления",
                                                key="apply_param_button")

        with column_2:
            # Ввод масштаба для изображений
            sb._center()
            self.input_scale_param = st.text_input("p", key="input_scale_param",
                                                   placeholder="Введите масштаб изображений",
                                                   label_visibility="collapsed")
            self._warning_input_scale_param()

        with column_3:
            # Радиокнопка для отключения расчета диаметра
            sb._center()
            self.radio_scale_button = st.radio("p",
                                               options=["Режим с измерениями",
                                                        "Режим без измерений"],
                                               label_visibility="collapsed",
                                               key="radio_scale_button")
            if self.radio_scale_button == "Режим без измерений":
                self.progress.info("Включен режим без измерений")

        with column_4:
            # Кнопка очистки папки temp
            sb._center()
            self.clean_temp_button = st.button("Очистка temp", key="clean_temp_button")

        # Очистка папки temp
        if self.clean_temp_button:
            # Очистка рабочей директории
            self._clean_temp()
            st.info("папка temp очищена")

    def _prepare_data(self):
        if self.uploaded_images is not None:
            # Время начала выполнения
            self.start_time = time.time()

            # Обрезка загруженного изображения пользователем
            self._separate_uploaded_image()

            if self.radio_scale_button == "Режим с измерениями":

                # Коробка с палкой
                self._separate_bboxstick()

                if self.input_scale_bool:
                    # Мини коробка с масштабом изображения
                    self._separate_minibbox()

                    # Извлекаем масштаб из минибокса
                    self._define_scale()

            # Контейнер содержащий визуализацию аннотированного изображения
            self.container_annotate_image()

    def _calculate_time_work(self):
        if self.uploaded_images is not None:
            # Время окончания выполнения функции
            self.end_time = time.time()
            # Вычисление времени выполнения
            self.execution_time = self.end_time - self.start_time
            # Вывод времени выполнения на Streamlit
            self.progress.success(f"Время работы алгоритма: {self.execution_time:.2f} сек")

    def _clean_temp(self):
        shutil.rmtree(self.path_to_uploaded_image)
        shutil.rmtree(self.path_to_main_image)
        shutil.rmtree(self.path_to_bbox)
        shutil.rmtree(self.path_to_minibbox)
        shutil.rmtree(self.path_to_bboxstick)

    def _make_os_dir(self):
        os.makedirs(self.path_to_uploaded_image, exist_ok=True)
        os.makedirs(self.path_to_main_image, exist_ok=True)
        os.makedirs(self.path_to_bbox, exist_ok=True)
        os.makedirs(self.path_to_minibbox, exist_ok=True)
        os.makedirs(self.path_to_bboxstick, exist_ok=True)
        os.makedirs(self.path_to_inference, exist_ok=True)

    def _load_images(self):
        """
        Загрузка изображений пользователя.
        """

        # Создаем контейнер с помощью st.container()
        self.cont_load_images = st.container()

        self.helper.create_indentations_in_container(1, self.cont_load_images)

        self.cont_load_images.divider()

        # Загрузка изображения через Streamlit
        self.uploaded_images = self.cont_load_images.file_uploader("Загрузите изображение",
                                                                   type=["jpg", "png", "jpeg", "tiff"],
                                                                   accept_multiple_files=True)

        self.cont_load_images.divider()

        if self.uploaded_images:
            for index, uploaded_image in enumerate(self.uploaded_images):
                self.name_uploaded_images.append(uploaded_image.name)
                self.name_uploaded_images_without_ex.append(os.path.splitext(uploaded_image.name)[0])
            if self.input_scale_bool:
                self.scale = np.ones(len(self.uploaded_images))
                self.bboxstick = np.ones(len(self.uploaded_images))
            else:
                if len(self.scale) != len(self.uploaded_images):
                    self.progress.error(f"Кол. значений scale: {len(self.scale)} не соответствует"
                                        f" кол. загруженных изображений count: {len(self.uploaded_images)}")
                    self.scale = []

        if self.uploaded_images is not None:
            # Создаем временную директорию для сохранения загруженного изображения
            os.makedirs(self.temp_dir, exist_ok=True)

            for index, image in enumerate(stqdm(self.uploaded_images,
                                                total=len(self.uploaded_images),
                                                desc="Загружаем картинки...",
                                                st_container=self.progress)):
                path_to_image = os.path.join(self.path_to_uploaded_image, f"{self.name_uploaded_images[index]}")
                image = Image.open(image)
                image = image.convert('RGB')
                image = image.resize(size=(512, 512))
                image.save(path_to_image)
                self.list_path_to_uploaded_images.append(path_to_image)

                if index == len(self.uploaded_images) and index != 0:
                    self.load_image_done = True

    def _separate_uploaded_image(self):
        for index, image in enumerate(stqdm(self.list_path_to_uploaded_images,
                                            total=len(self.list_path_to_uploaded_images),
                                            desc="Убираем колонтитул...",
                                            st_container=self.progress)):

            # Открываем загруженное изображение пользователем
            uploaded_image = Image.open(image)

            # Предсказание ограничивающей рамки для черного бокса
            self.results_detect_bbox = self.detect_model_bbox.predict(source=image,
                                                                      device=0,
                                                                      imgsz=512,
                                                                      save=False,
                                                                      conf=0.1,
                                                                      iou=0.1,
                                                                      verbose=False,
                                                                      augment=True)

            # Вычисляем координаты рамки для обрезки загруженного изображения пользователем
            detect_bbox = self.results_detect_bbox[0].boxes

            # Преобразование boxes_ в числовой тип данных и учет устройства
            self.boxes_bbox = detect_bbox.xywhn.cpu().numpy()

            for box in self.boxes_bbox:
                x, y, w, h = box

                # Умножение нормализованных координат на размеры исходного изображения
                x_pixel = int((x - w / 2) * 512)
                y_pixel = int((y - h / 2) * 512)
                w_pixel = int(w * 512)
                h_pixel = int(h * 512)

                # Обрезка изображения
                self.bbox = uploaded_image.crop((x_pixel, y_pixel, x_pixel + w_pixel, y_pixel + h_pixel))

            # Сохраняем основное изображение
            os.makedirs(self.temp_dir, exist_ok=True)
            path_to_bbox = os.path.join(self.path_to_bbox, f"{self.name_uploaded_images_without_ex[index]}.png")
            self.bbox = self.bbox.resize((1600, 80))
            self.bbox = self.bbox.convert("RGB")
            self.bbox.save(path_to_bbox)
            self.list_path_to_bbox.append(path_to_bbox)

            # Координаты и размеры второй части изображения
            x_main_image = 0
            y_main_image = 0
            w_main_image = 512
            h_main_image = 512 - h_pixel

            # Обрезка и получение основного изображения
            self.main_image = uploaded_image.crop((x_main_image, y_main_image, w_main_image, h_main_image))

            # Подготовка основного изображения к inference
            self.main_image = self.main_image.resize(size=(512, 512))

            # Сохраняем основное изображение
            os.makedirs(self.temp_dir, exist_ok=True)
            path_to_main_image = os.path.join(self.path_to_main_image,
                                              f"{self.name_uploaded_images_without_ex[index]}.png")
            self.main_image.save(path_to_main_image)
            self.list_path_to_main_image.append(path_to_main_image)

    def _separate_bboxstick(self):
        for index, image in enumerate(stqdm(self.list_path_to_bbox,
                                            total=len(self.list_path_to_bbox),
                                            desc="Измеряем длину палки...",
                                            st_container=self.progress)):

            # Открываем bbox
            bbox = Image.open(image)
            bbox = bbox.resize((512, 36))

            # Предсказание ограничивающей рамки для черного бокса
            self.results_detect_bboxstick = self.detect_model_bboxstick.predict(source=bbox,
                                                                                device=0,
                                                                                imgsz=512,
                                                                                save=False,
                                                                                conf=0.1,
                                                                                iou=0.1,
                                                                                verbose=False,
                                                                                augment=True)

            # Вычисляем координаты рамки для обрезки загруженного изображения пользователем
            detect_bboxstick = self.results_detect_bboxstick[0].boxes

            # Преобразование boxes_ в числовой тип данных и учет устройства
            self.boxes_bboxstick = detect_bboxstick.xywhn.cpu().numpy()

            for box in self.boxes_bboxstick:
                x, y, w, h = box

                # Умножение нормализованных координат на размеры исходного изображения
                x_pixel = int((x - w / 2) * 512)
                y_pixel = int((y - h / 2) * 36)
                w_pixel = int(w * 512)
                h_pixel = int(h * 36)

                # Обрезка изображения
                self.bboxstick = bbox.crop((x_pixel, y_pixel, x_pixel + w_pixel, y_pixel + h_pixel))

            # Сохраняем основное изображение
            os.makedirs(self.temp_dir, exist_ok=True)
            path_to_bboxstick = os.path.join(self.path_to_bboxstick,
                                             f"{self.name_uploaded_images_without_ex[index]}.png")
            self.bboxstick = self.bboxstick.convert("RGB")
            self.bboxstick.save(path_to_bboxstick)
            self.list_path_to_bboxstick.append(path_to_bboxstick)

            width, _ = self.bboxstick.size
            self.stick.append(width)
        st.info(f"stick: {self.stick}")

    def _separate_minibbox(self):
        for index, bbox in enumerate(stqdm(self.list_path_to_bbox,
                                           total=len(self.list_path_to_bbox),
                                           desc="Извлекаем рамочку...",
                                           st_container=self.progress)):

            bbox_ = Image.open(bbox)

            # Предсказание ограничивающей рамки для черного бокса
            self.results_detect_minibbox = self.detect_model_minibbox.predict(source=bbox,
                                                                              device=0,
                                                                              imgsz=512,
                                                                              save=False,
                                                                              conf=0.1,
                                                                              iou=0.1,
                                                                              verbose=False)

            # Вычисляем координаты рамки для обрезки загруженного изображения пользователем
            detect_minibbox = self.results_detect_minibbox[0].boxes

            # Преобразование boxes_ в числовой тип данных и учет устройства
            self.boxes_minibbox = detect_minibbox.xywhn.cpu().numpy()

            for box in self.boxes_minibbox:
                x, y, w, h = box

                # Умножение нормализованных координат на размеры исходного изображения
                x_pixel = int((x - w / 2) * 1600)
                y_pixel = int((y - h / 2) * 80)
                w_pixel = int(w * 1600)
                h_pixel = int(h * 80)

                # Обрезка изображения
                self.minibbox = bbox_.crop((x_pixel, y_pixel, x_pixel + w_pixel, y_pixel + h_pixel))

            path_to_minibbox = os.path.join(self.path_to_minibbox,
                                            f"{self.name_uploaded_images_without_ex[index]}.png")
            self.minibbox = self.minibbox.convert('L')
            self.minibbox = self.minibbox.resize((90, 40))
            self.minibbox.save(path_to_minibbox)
            self.list_path_to_minibbox.append(path_to_minibbox)

    def _define_scale(self):
        for index, image in enumerate(stqdm(self.list_path_to_minibbox,
                                            total=len(self.list_path_to_minibbox),
                                            desc="Получаем масштаб...",
                                            st_container=self.progress)):

            # Определяем масштаб, извлекая текст из мини бокса
            results = self.reader.readtext(image, detail=0)
            result = results[0].replace(" ", "")
            # st.info(f"res: {results}")

            # Убираем букву "m" из строки
            result_without_m = result.replace("m", "")
            # st.info(f"res: {result_without_m}")

            # Ищем позицию буквы "u" или "n"
            index_of_u = result_without_m.find("u")
            index_of_n = result_without_m.find("n")

            # Определяем индекс символа, который встречается раньше
            if index_of_u != -1 and index_of_n != -1:
                index_of_symbol = min(index_of_u, index_of_n)
            elif index_of_u != -1:
                index_of_symbol = index_of_u
                self.scale[index] *= 1000
            elif index_of_n != -1:
                index_of_symbol = index_of_n
                self.scale[index] *= 1
            else:
                # Если ни "u", ни "n" не найдены, пропускаем обработку
                self.scale[index] = 0

            # st.info(f"index_of_symbol: {index_of_symbol}")
            # Вычисляем число перед символом "u" или "n"
            number_before_u_or_n = int(result_without_m[:index_of_symbol])
            # st.info(f"number_before_u_or_n: {number_before_u_or_n}")
            self.scale[index] *= number_before_u_or_n
            st.info(f"scale: {self.scale[index]}")

    def container_annotate_image(self):
        """
        Контейнер, содержащий визуализацию аннотированного изображения.
        """

        for index, (image, scale) in enumerate(
                stqdm(zip(self.list_path_to_main_image, self.scale),
                      total=len(self.list_path_to_main_image),
                      desc="Считаем...",
                      st_container=self.progress)):

            image_ = Image.open(image)

            if self.path_to_detect_model and self.path_to_segment_model is not None:

                self.results_detect = self.detect_model.predict(source=image,
                                                                device=0,
                                                                imgsz=512,
                                                                save=False,
                                                                conf=0.1,
                                                                iou=0.1,
                                                                verbose=False)

                detect = self.results_detect[0].boxes

                # Преобразование boxes_ в числовой тип данных и учет устройства
                self.boxes_ = detect.xywhn.cpu().numpy()

                # Для сохранения размеров изображения детекций
                roi_size_image = []

                for box in self.boxes_:
                    x, y, w, h = box

                    # Умножение нормализованных координат на размеры исходного изображения
                    x_pixel = int((x - w / 2) * 512)
                    y_pixel = int((y - h / 2) * 512)
                    w_pixel = int(w * 512)
                    h_pixel = int(h * 512)

                    # Обрезка изображения
                    roi = image_.crop((x_pixel, y_pixel, x_pixel + w_pixel, y_pixel + h_pixel))

                    roi_size_image.append(roi.size)

                    # roi = roi.resize(size=(256, 256))
                    roi = roi.resize(size=(224, 224))

                    self.segmented_images.append(roi)

                for i, seg_image in enumerate(self.segmented_images):
                    # Преобразование изображения в массив NumPy
                    img_array = img_to_array(seg_image)

                    # Предсказание сегментации с помощью модели
                    mask_pred = predict(model=self.segment_model, inp=img_array)

                    # Преобразование маски в бинарную маску (0 или 1)
                    binary_mask = (mask_pred > 0.7).astype(np.uint8)

                    # Преобразование бинарной маски в объект PIL Image
                    mask_pred_pil = Image.fromarray(binary_mask * 255)

                    # Добавление результата в список результатов сегментации
                    self.segmented_results.append(mask_pred_pil)

                # Создаем копию исходного изображения
                image_copy_with_masks = image_.convert("RGBA")

                # Проходим по каждой маске в segmented_results
                for i, mask_pred_pil in enumerate(self.segmented_results):
                    # Преобразование маски в формат RGBA
                    mask_rgba = mask_pred_pil.convert("RGBA")

                    # Проходим по результатам детекции и получаем соответствующий бокс
                    box = self.boxes_[i]
                    x, y, w, h = box

                    # Умножение нормализованных координат на размеры исходного изображения
                    x_pixel = int((x - w / 2) * 512)
                    y_pixel = int((y - h / 2) * 512)
                    w_pixel = int(w * 512)
                    h_pixel = int(h * 512)

                    # Обрезаем маску до размеров бокса
                    mask_cropped = mask_rgba.resize(size=(w_pixel, h_pixel))

                    # Преобразование массива NumPy
                    mask_array = np.array(mask_cropped)

                    # Заменяем черный цвет на прозрачный (alpha=0)
                    mask_array[(mask_array[:, :, :3] == [0, 0, 0]).all(axis=2), 3] = 0

                    # Заменяем белый цвет на синий (RGB=[0, 0, 255])
                    mask_array[(mask_array[:, :, :3] > [20, 20, 20]).all(axis=2), :3] = [0, 0, 255]
                    mask_array[(mask_array[:, :, :3] == [0, 0, 255]).all(axis=2), 3] = 60

                    # Создаем Image из массива NumPy
                    mask_cropped = Image.fromarray(mask_array)

                    # Наложение маски на изображение внутри бокса
                    image_copy_with_masks.paste(mask_cropped, (x_pixel, y_pixel), mask_cropped)

                if self.radio_scale_button == "Режим с измерениями":
                    diameters = self.calculate_particle_diameter(scale, self.stick[index], roi_size_image)

                # Создаем объект ImageDraw для рисования на изображении
                draw = ImageDraw.Draw(image_copy_with_masks)

                # Проходим по результатам детекции и рисуем прямоугольники
                count = 0
                for i, box in enumerate(self.boxes_):
                    count += 1
                    x, y, w, h = box

                    # Умножение нормализованных координат на размеры исходного изображения
                    x_pixel = int((x - w / 2) * 512)
                    y_pixel = int((y - h / 2) * 512)
                    w_pixel = int(w * 512)
                    h_pixel = int(h * 512)

                    # Рисуем прямоугольник
                    draw.rectangle([x_pixel, y_pixel, x_pixel + w_pixel, y_pixel + h_pixel], outline="green",
                                   width=1)

                    add_text_to_image(draw=draw,
                                      text=f"{i}",
                                      position=[x_pixel + 2, y_pixel],
                                      outline_width=0)

                    if self.radio_scale_button == "Режим с измерениями":
                        add_text_to_image(draw=draw,
                                          text=f"{diameters[i]: .0f}nm",
                                          position=[x_pixel - 4, y_pixel + (h_pixel - 10)],
                                          outline_width=0)

                # Запись количества частиц на одном изображении в словарь.
                self.dir_num_sphere_for_every_image[self.name_uploaded_images_without_ex[index]] = count
                if self.radio_scale_button == "Режим с измерениями":
                    # Запись значения диаметров частиц на одном изображении в словарь.
                    self.dir_diameters_for_every_image[self.name_uploaded_images_without_ex[index]] = diameters
                    self.dir_scales_for_every_image[self.name_uploaded_images_without_ex[index]] = scale

                # Сохраняем изображение в inference
                if image_copy_with_masks is not None:
                    self.path_to_inference_dir = os.path.join(self.path_to_inference, f"{self.timestamp}")
                    self.path_to_inference_dir_image = os.path.join(self.path_to_inference_dir, "image")
                    self.path_to_inference_dir_boxplot = os.path.join(self.path_to_inference_dir, "boxplot")
                    self.path_to_inference_dir_json = os.path.join(self.path_to_inference_dir, "json")
                    os.makedirs(self.path_to_inference_dir, exist_ok=True)
                    os.makedirs(self.path_to_inference_dir_image, exist_ok=True)
                    os.makedirs(self.path_to_inference_dir_boxplot, exist_ok=True)
                    os.makedirs(self.path_to_inference_dir_json, exist_ok=True)
                    path_to_image_copy_with_mask = os.path.join(self.path_to_inference_dir_image,
                                                                f"{self.name_uploaded_images_without_ex[index]}.png")
                    image_copy_with_masks.save(path_to_image_copy_with_mask)
                    self.list_path_to_inference.append(path_to_image_copy_with_mask)
                    self.list_image_copy_with_mask.append(image_copy_with_masks)

            # Добавление количества частиц для каждого изображения в data
            self.dir_data = self.add_key_value(self.dir_data, "num_particles", self.dir_num_sphere_for_every_image)
            if self.radio_scale_button == "Режим с измерениями":
                # Добавление вычисленных значений диаметров для каждого изображения в data
                self.dir_data = self.add_key_value(self.dir_data, "diameters", self.dir_diameters_for_every_image)
                # Добавление значений масштаба для каждого изображения в data
                self.dir_data = self.add_key_value(self.dir_data, "scales", self.dir_scales_for_every_image)

            path_to_json_file = os.path.join(self.path_to_inference_dir_json, "info.json")

            # Сохранение JSON-файла
            self.save_to_json(data=self.dir_data, file_path=path_to_json_file)

            # После каждого предсказания, необходимо обнулять переменные, хранящие результаты работы моделей
            self._refresh_inference_params()

    def mask_to_polygon(self, mask_array):
        """
        Преобразует бинарную маску в полигон.

        Параметры:
        - mask_array (numpy.ndarray): Бинарная маска (0 и 255).

        Возвращает:
        - List[tuple]: Список координат точек полигона.
        """
        contours, _ = cv2.findContours(mask_array.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:  # Проверяем, есть ли контуры
            return []  # Возвращаем пустой список, если контуры не найдены

        # Выбираем контур с наибольшей площадью (если есть несколько)
        largest_contour = max(contours, key=cv2.contourArea)

        # Преобразуем контур в список координат точек полигона
        polygon = [tuple(point[0]) for point in largest_contour]

        return polygon

    def calculate_particle_diameter(self, scale, stick, roi_size):
        """
        Вычисляет диаметр сферических наночастиц по сегментам.

        Параметры:
        - scale (float): Масштаб изображения.

        Возвращает:
        - List[float]: Список диаметров частиц в реальных единицах длины.
        """
        particle_diameters = []

        for i, mask_pred_pil in enumerate(self.segmented_results):
            # Проходим по результатам детекции и получаем соответствующий бокс
            box = self.boxes_[i]
            x, y, w, h = box

            # Умножение нормализованных координат на размеры исходного изображения
            x_pixel = int((x - w / 2) * 512)
            y_pixel = int((y - h / 2) * 512)
            w_pixel = int(w * 512)
            h_pixel = int(h * 512)

            # Обрезаем маску до размеров бокса
            mask_pred_pil = mask_pred_pil.resize(size=(w_pixel, h_pixel))

            # Получаем массив NumPy из маски
            mask_array = np.array(mask_pred_pil) / 255

            # Получаем полигон из бинарной маски
            polygon = self.mask_to_polygon(mask_array)

            def feret_diameter(polygon):
                diameter_array = []
                for i in range(len(polygon)):
                    for j in range(i + 1, len(polygon)):
                        # Вычисляем длину отрезка между двумя точками
                        diameter = ((polygon[i][0] - polygon[j][0]) ** 2 + (polygon[i][1] - polygon[j][1]) ** 2) ** 0.5
                        diameter_array.append(diameter)

                diameter = np.mean(diameter_array[round(len(diameter_array) / 1.5):])
                # diameter = np.mean([np.median(diameter_array), np.max(diameter_array)])

                return diameter

            diameter = feret_diameter(polygon)

            # Умножаем на масштаб для получения реального диаметра в физических единицах
            particle_diameter = diameter * (scale / stick)  # Перевод нанометров в микрометры

            particle_diameters.append(particle_diameter)

        return particle_diameters

    def _boxplot_num_sphere_all_image(self):

        if self.path_to_inference_dir_boxplot:

            if self.uploaded_images is not None:
                # Инициализация списка для хранения значений количества частиц для каждого файла
                all_counts = []

                # Извлекаем из словаря значения количества частиц для каждого изображения
                for filename, count in self.dir_num_sphere_for_every_image.items():
                    all_counts.append(count)

                # Сохраняем боксплот
                plot_boxplot = BOXPLOT(path=self.path_to_inference_dir_boxplot,
                                       title="Диаграмма размаха количества частиц",
                                       total=all_counts)

                self.helper.create_indentations_in_container(3, self.cont_load_images)
