import streamlit as st
from library.IndentationHelper import IndentationHelper
from library.components import footer, footer_style


class PageInfo:

    def __init__(self):
        # Это используется для скрытия предупреждения о кодировании при загрузке файла.
        st.set_option('deprecation.showfileUploaderEncoding', False)

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
        st.markdown(
            "<h3 style='text-align: center; color: #00DDFA;'>Добро пожаловать в приложение MicroscopeAI</h1>",
            unsafe_allow_html=True)

        # st.markdown('')
        # st.markdown('**Overview**')
        # st.markdown(
        # '<div style="text-align: justify;">TFinder is an easy-to-use Python web portal allowing the identification of Individual Motifs (IM) such as Transcription Factor Binding Sites (TFBS). Using the NCBI API, TFinder extracts either promoter or the gene terminal regions through a simple query based on NCBI gene name or ID. It enables simultaneous analysis across five different species for an unlimited number of genes. TFinder searches for TFBS and IM in different formats, including IUPAC codes and JASPAR entries. Moreover, TFinder also allows the generation and use of a Position Weight Matrix (PWM). Finally, the data can be recovered in a tabular form and a graph showing the relevance of the TFBSs and IMs as well as its location relative to the Transcription Start Site (TSS) or gene end. The results are then sent by email to the user facilitating the subsequent analysis and data analysis sharing.</div>',
        # unsafe_allow_html=True)
        # st.divider()
        # st.markdown(
        # '<div style="text-align: justify;"><p style="text-indent: 2em;"></p></div>',
        # unsafe_allow_html=True)

        st.markdown(
            '<div style="text-align: justify;"><p style="text-indent: 2em;">Это приложение предназначено для классификации и сегментации изображений сканирующей электронной микроскопии (СЭМ).</p></div>',
            unsafe_allow_html=True)
