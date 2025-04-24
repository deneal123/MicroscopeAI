<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=100&size=32&pause=1000&center=true&vCenter=true&multiline=true&repeat=false&random=false&width=950&lines=MicroscopeAI" alt="Typing SVG" /></a>

---

Приложение (локальный-интерфейс) предназначенное для измерения размеров наночастиц сферической формы.
В настоящее время находится в разработке.


> [**Разработка алгоритма классификации и сегментации
> СЭМ-изображений сферических наночастиц на поверхности биосовместимых
> материалов**](https://cyberleninka.ru/article/n/razrabotka-algoritma-klassifikatsii-i-segmentatsii-sem-izobrazheniy-sfericheskih-nanochastits-na-poverhnosti-biosovmestimyh)
> Вольхин Д.Ф - НИУ МИЭТ // Микроэлектроника и информатика 2024.


### Установка

1. Клонируйте репозиторий
или скачайте [репозиторий](https://github.com/deneal123/MicroscopeAI/archive/refs/heads/master.zip)
и распакуйте в удобное место.
```
git clone https://github.com/deneal123/MicroscopyAI.git
```
2. Запустите файл `setup.but` и выберите вариант "*Установка MicroscopeAI*".
3. Скачайте и распакуйте [архив](https://disk.yandex.ru/d/) `cudnn_windows` в корень репозитория 
4. После установки всех необходимых зависимостей установите файлы cudnn, выбрав вариант "*Установка cudnn файлов*"
в `setup.bat`.
5. Скачайте [модели детекции](https://disk.yandex.ru/d/) и
[модель сегментации](https://disk.yandex.ru/d/), распаковав их по путям `weights/weights_detect`
и `weights/weights_seg` в корень директории репозитория соответственно.
6. Для запуска приложения можно использовать соответсвующий пункт '*Запуск Web-UI в браузере*'
в `setup.bat` или запустить `webui.bat`.



### Структура репозитория





### Дополнительно

Для работы приложения необходимо наличие весов соответствующих моделей, которые могут быть заменены,
в указанных директориях:

- "/weights/weights_detect/best_detect_3.pt" `Yolov8x`
- "/weights/weights_detect/best_detect_bbox.pt" `Yolov8m`
- "/weights/weights_detect/best_detect_bboxstick.pt" `Yolov8m`
- "/weights/weights_detect/best_detect_minibbox.pt" `Yolov8m` 
- "/weights/weights_seg/" `UEffifientNetB3` *Архитектура модели в библиотеке ./library*


Если хотите использовать другие веса, необходимо модернизировать функцию `default_path()`
в скрипте `config_file.py`, заменив соответствующие названия весов.



### Демо

![Пример работы алгоритма](/img/combined_image_res_algoritm.jpg)


### Как работает приложение?

![Блокс схема алгоритма](/img/BlokShema.svg)


### Лицензия

Этот проект распространяется под лицензией [GPLv3](LICENSE).

### Цитирование

    @article{2025nanoparticles,
    title={РАЗРАБОТКА АЛГОРИТМА КЛАССИФИКАЦИИ И СЕГМЕНТАЦИИ СЭМ-ИЗОБРАЖЕНИЙ СФЕРИЧЕСКИХ НАНОЧАСТИЦ НА ПОВЕРХНОСТИ БИОСОВМЕСТИМЫХ МАТЕРИАЛОВ},
    author={Рябкин, Д. И. and Вольхин, Д. Ф.},
    journal={Экономика и качество систем связи},
    number={35},
    pages={101--110},
    year={2025},
    url={https://cyberleninka.ru/article/n/razrabotka-algoritma-klassifikatsii-i-segmentatsii-sem-izobrazheniy-sfericheskih-nanochastits-na-poverhnosti-biosovmestimyh},
    }

