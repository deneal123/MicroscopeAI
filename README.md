<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=100&size=32&pause=1000&center=true&vCenter=true&multiline=true&repeat=false&random=false&width=950&lines=MicroscopeAI" alt="Typing SVG" /></a>

---

Приложение (веб-интерфейс) предназначенное для измерения размеров наночастиц сферической формы.
В настоящее время находится в разработке. С помощью различных методов сгенерирован синтетический набор данных, на котором обучены модель детекции Yolov8x и модель сегментации Unet с бэкбоном senet154.

#### .EasyOcr положить в папку model - minibbox_ocr.pth, в папку user_network - minibbox_ocr.py и minibbox_ocr.yaml
#### .Streamlit положить файл config

### Установка

1. Клонируйте репозиторий: `git clone https://github.com/deneal123/MicroscopyAI.git`
2. Запустите файл `setup.but` и выберите нужный вариант установки. `Черное окно? Перезапустите`
3. Дождитесь установки всех компонентов, затем запустите файл `webui.bat`

### Интерфейс

- **webui_train**: обучение моделей классификации и сегментации
- **webui_demo**: измерение частиц

### Дополнительно

Для работы приложения необходимо скачать файлы весов соответствующих моделей и расположить их в следующие папки в корне репозитория:

- "/weights/weights_detect/best_detect_2.pt"
- "/weights/weights_detect/best_detect_bbox.pt"
- "/weights/weights_detect/best_detect_bboxstick.pt"
- "/weights/weights_detect/best_detect_minibbox.pt"
- "/weights/weights_seg/2024_03_20_23_16_00_UNet_se_resnet50_micronet.pth"

А также установить папку cudnn_windows

[ссылка на скачивание](https://disk.yandex.ru/d/aJDHGdLKqtZVLw)


### Демо

![Интерфейс обучения](https://github.com/deneal123/MicroscopyAI/blob/master/img/train.png)

![Интерфейс Demo](https://github.com/deneal123/MicroscopyAI/blob/master/img/demo.png)

### Лицензия

Этот проект распространяется под лицензией [AGPL-3.0 License](LICENSE).
