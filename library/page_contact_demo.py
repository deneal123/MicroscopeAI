import streamlit as st
from library.custom_logging import setup_logging
from library.IndentationHelper import IndentationHelper
from config_file import load_config, is_value_changed, save_config
from PIL import Image


class PageContact:

    def __init__(self):
        # Это используется для скрытия предупреждения о кодировании при загрузке файла.
        st.set_option('deprecation.showfileUploaderEncoding', False)

        # Назначение переменной логирования
        self.log = setup_logging()

        # Загрузка конфигурационного файла
        self.config_data = load_config()
        self.dir_mapping = self.config_data["dir_mapping"]

        # Класс функций для отступов
        self.helper = IndentationHelper()

    def run(self):
        """
        Запуск приложения.
        """
        st.divider()

        # Содержимое страницы
        self.title_page()

    def title_page(self):
        """
        Содержимое страницы ввиде вступительного текста.
        """

    def update_config(self):
        """
        Обновляет конфигурационные данные на основе переданных параметров.
        """

        save_config(self.config_data)