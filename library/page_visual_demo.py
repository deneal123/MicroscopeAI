import os
from library.style_button import _center, _move_button
import streamlit as st
from library.custom_logging import setup_logging
from library.IndentationHelper import IndentationHelper
from config_file import load_config, is_value_changed, save_config
from PIL import Image
import json


class PageContact:

    def __init__(self):
        # Назначение переменной логирования
        self.log = setup_logging()

        # Загрузка конфигурационного файла
        self.config_data = load_config()

        # Класс функций для отступов
        self.helper = IndentationHelper()

        self.path_to_inference = self.config_data["path_to_inference"]

        self.path_to_json = ""

        self.list = []

        self.list_dit = []

        self.list_image = []

        self.name_list = []

        self.choice_dir_selectbox = None

        self.path_to_inference_time = ""

        self.num_particles, self.diameters, self.scales = {}, {}, {}

    def run(self):
        """
        Запуск приложения.
        """
        st.divider()

        # Содержимое страницы
        self.title_page()

        self._get_path_list_to_image()

        self.container_params()

        if self.choice_dir_selectbox != "Не указано":
            # Визуализация аннотированных изображений
            self._show()

    def title_page(self):
        """
        Содержимое страницы ввиде вступительного текста.
        """
        self.helper.create_indentations(1)
        self.progress = st.container()

    def container_params(self):

        self.cont_params = st.container()

        # Создание двух столбцов в контейнере
        column_1, column_2, column_3 = self.cont_params.columns(3)

        with column_1:
            # Выбор директории для визуализации
            _center()
            self.choice_dir_selectbox = st.selectbox("Выберите данные для отображения",
                                                     self.list_dir)

            self.path_to_inference_time = os.path.join(self.path_to_inference, self.choice_dir_selectbox)

            self.path_to_json = os.path.join(self.path_to_inference_time, "json", "info.json")

        with column_2:
            pass

        with column_3:
            # Кнопка для выбора масштаба визуализации изображения
            _center()
            self.input_scale_image = st.slider("Введите масштаб визуализации",
                                               value=832,
                                               min_value=256,
                                               max_value=832,
                                               step=32,
                                               key="input_scale_image_param")

    def _get_path_list_to_image(self):

        self.list_dir = os.listdir(self.path_to_inference)
        self.list_dir.insert(0, "Не указано")

    def _show(self):

        # Контейнер для показа изображений
        self.cont_show_image = st.container()

        self.helper.create_indentations_in_container(3, self.cont_show_image)

        try:

            # Получаем словари из json файла
            self.num_particles, self.diameters, self.scales = read_json_file(self.path_to_json)

            self.list = os.listdir(os.path.join(self.path_to_inference_time, "image"))

            self.name_list = [os.path.splitext(filename)[0] for filename in self.list]

            self.list = [os.path.join(self.path_to_inference_time, "image", f) for f in self.list]

            for image in self.list:
                image = Image.open(image)
                self.list_image.append(image)

            scale = None

            if len(self.list_image) == 1:

                try:
                    scale = self.scales[f"{self.name_list[0]}"]
                except KeyError as ke:
                    pass

                num_particles = self.num_particles[f"{self.name_list[0]}"]

                if scale is None:
                    caption = f'Имя: {self.name_list[0]}, кол.частиц: {num_particles}'
                else:
                    caption = f'Имя: {self.name_list[0]}, масштаб: {scale}нм, кол.частиц: {num_particles}'

                # Отображаем изображение с наложенными масками
                self.cont_show_image.image(self.list_image[0],
                                           caption=caption,
                                           use_column_width=False,
                                           width=self.input_scale_image)

            else:

                k_slider = self.cont_show_image.slider("Выберите изображение для отображения",
                                                       min_value=1,
                                                       max_value=len(self.list_image),
                                                       value=1,
                                                       step=1,
                                                       key="slider")

                try:
                    scale = self.scales[f"{self.name_list[k_slider - 1]}"]
                except KeyError as ke:
                    pass

                num_particles = self.num_particles[f"{self.name_list[k_slider - 1]}"]

                if scale is None:
                    caption = (f'Имя: {self.name_list[k_slider - 1]},'
                               f' кол.частиц: {num_particles}')
                else:
                    caption = (f'Имя: {self.name_list[k_slider - 1]},'
                               f' масштаб: {scale}нм,'
                               f' кол.частиц: {num_particles}')

                # Отображаем изображение с наложенными масками
                self.cont_show_image.image(self.list_image[k_slider - 1],
                                           caption=caption,
                                           use_column_width=False,
                                           width=self.input_scale_image)

        except FileExistsError as ex:
            st.info("Директория повреждена")


def read_json_file(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

    num_particles = data.get('num_particles', {})
    diameters = data.get('diameters', {})
    scales = data.get('scales', {})

    return num_particles, diameters, scales
