import os
from PIL import Image, ImageDraw, ImageFont
import string
import streamlit as st

footer_style = f"""
    <style>
        MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        footer:after {{ 
            visibility: visible;
            display: block;
            position: relative;
            # background-color: red;
            padding: 5px;
            top: 2px;
        }}
    </style>
"""

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """

icons = """<style>
        img {
            max-width: 100%;

        }

        .headerStyle {
            transition: transform .2s;
        }

        .headerStyle:hover {

             transform: scale(1.5);
            transition: 0.2s;
        }
        .image1 { 
            padding: 10px;
             transition: transform .2s;
        }
        .image2 
        { 
            padding: 10px;
             transition: transform .2s;
        }
        .image1:hover {  
            ##border: 4px solid green;
            ##border-radius: 15px;
             transform: scale(1.5);
            transition: 0.2s;
        }

        .image2:hover {  
            ##border: 4px solid green;
            ##border-radius: 15px;
             transform: scale(1.5);
            transition: 0.2s;
        }

        a:link,
        a:visited {
            color: #00DDFA;
            background-color: transparent;
            text-decoration: underline;
        }

        a2:hover {
            border-style: solid;
            },
        a:active {
            color: red;
            background-color: transparent;
            text-decoration: underline;
        }

        .footer {
            position: fixed;
            width: 100%;
            background-color: black;
            color: #00DDFA;
            display: block;
            text-align: center;
            padding: 100px;
            top: 0px;
        }
</style>
"""

footer = """<style>
img {
            max-width: 100%;

        }

        .headerStyle {
            transition: transform .2s;
        }

        .headerStyle:hover {

             transform: scale(1.5);
            transition: 0.2s;
        }
        .image1 { 
            padding: 10px;
             transition: transform .2s;
        }
        .image2 
        { 
            padding: 10px;
             transition: transform .2s;
        }
        .image1:hover {  
            ##border: 4px solid green;
            ##border-radius: 15px;
             transform: scale(1.5);
            transition: 0.2s;
        }

        .image2:hover {  
            ##border: 4px solid green;
            ##border-radius: 15px;
             transform: scale(1.5);
            transition: 0.2s;
        }

        a:link,
        a:visited {
            color: red;
            background-color: transparent;
            text-decoration: underline;
        }

        a2:hover {
            border-style: solid;
            },
        a:active {
            color: red;
            background-color: transparent;
            text-decoration: underline;
        }
.footer {
position: relative;
width: 100%;
left: 0;
bottom: 0;
background-color: transparent;
margin-top: auto;
color: #00DDFA;
padding: 24px;
text-align: center;
}
</style>
<div class="footer">
<p style="font-size:  13px">Copyright (c) 2023, MicroscopeAI inc.</p>
<p style="font-size: 13px">This software is distributed under an ? licence. Please consult the LICENSE file for more details.</p>
</a><a href="https://github.com/deneal123"><img class="image2" src="https://cdn-icons-png.flaticon.com/512/25/25231.png" alt="github" width="70" height="70"></a>
"""

color_style = """
    <style>
    :root {
      --primary-color: black;
    }
    </style>
"""


def add_text_to_image(image, text, position, font_size=30, text_color=(255, 255, 0), outline_color=(0, 0, 0),
                      outline_width=2):
    # Создаем объект ImageDraw
    draw = ImageDraw.Draw(image)

    # Загружаем шрифт Arial
    script_path = os.path.realpath(__file__)
    font_path = os.path.join(script_path, "image/ArialCyr.ttf")
    font = ImageFont.truetype(font_path, font_size)

    # Рисуем текст на изображении с обводкой
    draw.text(position, text, font=font, fill=text_color, stroke_fill=outline_color, stroke_width=outline_width)
