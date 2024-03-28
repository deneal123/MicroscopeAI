import json
import statistics
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import sem, t


class BOXPLOT:

    def __init__(self,
                 path: str = "",
                 title: str = "",
                 size_window: tuple = (4, 6),
                 dpi: int = 500,
                 total: bytearray = None,
                 confidence_level: float = 0.95):
        """
        Description
        -------
        Построение, сохранение графика BOXPLOT и JSON-файла со значениями.
        Неизменный вид графика с медианной и средней линией.

        Methods
        -------
        __init__: Определяет переменные.
        _time: Получает текущее время.
        _mean_std_conf: Рассчитывает среднее значение, среднеквадратичную ошибку и доверительный интервал.
        _plot_boxplot: Строит график.
        _write_json_boxplot: Сохраняет JSON-файл.

        Params
        -------
        :param path: Путь сохранения графика, обязательная переменная. [str]
        :param title: Название графика, по умолчанию пустое. [str]
        :param size_window: Размер окна, по умолчанию (4, 6). [tuple]
        :param dpi: Разрешение графика, по умолчанию 500. [int]
        :param total: Массив данных, обязательная переменная. [bytearray]
        :param confidence_level: Уровень доверия, по умолчанию 0.95 [float]

        :return: Метод all_values возвращает рассчитанные данные.
        """

        try:
            self.fig_path = path
            self.title = title
            self.size_window = size_window
            self.total = list(map(float, total))
            self.dpi = dpi
        except TypeError:
            print("\033[91mОШИБКА: Не правильный тип входных данных\033[0m")

        self.confidence_level = confidence_level
        self._mean: float = 0.0
        self._std_error: float = 0.0
        self._confidence_interval: tuple = (0, 0)
        self._median: float = 0.0
        self._q1: float = 0.0
        self._q3: float = 0.0
        self._whisker_low: float = 0.0
        self._whisker_high: float = 0.0
        self._time: str = datetime.now().strftime("%Y_%m_%d_%H_%M")

        self.boxplot = None
        self.fig = None
        self.abs_path: str = ''

        # Вызываем функции построения графика и записи JSON-файла
        self._plot_boxplot()
        self._write_json_boxplot()

    @property
    def all_value(self):
        return (
            self._mean,
            self._std_error,
            self._confidence_interval,
            self._median,
            self._q1,
            self._q3,
            self._whisker_low,
            self._whisker_high
        )

    @property
    def _boxplot_(self):
        return self.abs_path

    def __str__(self):
        try:
            if (self.fig_path, self.title, self.size_window, self.total, self.dpi) is not None:
                return (f"-------------------------------------------------------\n"
                        f"Время создания BOXPLOT: {self._time}\n"
                        f"Путь к графику: {self.fig_path}\n"
                        f"Название графика: {self.title}\n"
                        f"Размер окна: {self.size_window}\n"
                        f"Разрешение графика: {self.dpi}\n"
                        f"Среднее значение: {self.all_value[0]: .4f}\n"
                        f"Уровень доверия: {self.confidence_level: .2f}\n"
                        f"Доверительный интервал: ({self.all_value[2][0]: .4f}, {self.all_value[2][1]: .4f})\n"  # Исправлено здесь
                        f"Медиана: {self.all_value[3]: .4f}\n"
                        f"Квантиль 1: {self.all_value[4]: .4f}\n"
                        f"Квантиль 3: {self.all_value[5]: .4f}\n"
                        f"Нижний ус: {self.all_value[6]: .4f}\n"
                        f"Верхний ус: {self.all_value[7]: .4f}\n"
                        f"-------------------------------------------------------\n")
        except AttributeError:
            pass
        return ""

    def _mean_std_conf(self):
        """
        Рассчитывает среднее значение, среднеквадратичную ошибку и доверительный интервал.
        """

        # Вычисление среднего значения
        self._mean = np.mean(self.total)
        # Вычисление стандартного отклонения (погрешности)
        self._std_error = sem(self.total)
        # Вычисление доверительного интервала
        margin_of_error = self._std_error * t.ppf((1 + self.confidence_level) / 2, len(self.total) - 1)
        self._confidence_interval = (self._mean - margin_of_error, self._mean + margin_of_error)

    def _plot_boxplot(self):
        """
        Строит боксплот.
        """

        # Определяем фигуру и размер окна
        self.fig = plt.figure(figsize=(self.size_window[0], self.size_window[1]))

        # Определяем название графика
        plt.title(self.title)

        # Строим боксплот
        self.boxplot = plt.boxplot(self.total,
                                   meanline=True,
                                   showmeans=True,
                                   showcaps=True,
                                   meanprops={'color': 'red', 'label': 'Среднее'},
                                   medianprops={'color': 'green', 'label': 'Медиана'})

        # Получение значений медианы, границ ящика и усов
        self._mean = self.boxplot['means'][0].get_ydata()[0]
        self._median = self.boxplot['medians'][0].get_ydata()[0]
        self._q1 = self.boxplot['boxes'][0].get_ydata()[0]
        self._q3 = self.boxplot['boxes'][0].get_ydata()[2]
        self._whisker_low = self.boxplot['whiskers'][0].get_ydata()[1]
        self._whisker_high = self.boxplot['whiskers'][1].get_ydata()[1]

        # Добавление численных значений на график
        plt.text(0.80, self._median, f'{self._median:.4f}', ha='center', va='center', color='green')
        plt.text(1.20, self._mean, f'{self._mean:.4f}', ha='center', va='center', color='red')
        plt.text(1.20, self._q1, f'{self._q1:.4f}', ha='center', va='center', color='black')
        plt.text(1.20, self._q3, f'{self._q3:.4f}', ha='center', va='center', color='black')
        plt.text(1.20, self._whisker_low, f'{self._whisker_low:.4f}', ha='center', va='center', color='black')
        plt.text(1.20, self._whisker_high, f'{self._whisker_high:.4f}', ha='center', va='center', color='black')

        # Вывод легенды для медианы и средней линии
        plt.legend(loc='upper left')

        self.abs_path = os.path.join(self.fig_path, f"{self._time}_BoxPlot.png")
        # Сохранение графика в отдельный файл
        plt.savefig(self.abs_path, dpi=self.dpi)

        # Закрытие графика
        plt.close()

    def _write_json_boxplot(self):

        # Определяем путь сохранения JSON-файла
        json_filename = os.path.join(self.fig_path,
                                     f"{self._time}_BoxPlot.json")

        # Определяем словарь для записи в JSON-файл
        result = {
            "Mean": self._mean,
            "Std_error": self._std_error,
            "Confidence_interval": self._confidence_interval,
            "Meadian": self._median,
            "Q1": self._q1,
            "Q3": self._q3,
            "Whisker_low": self._whisker_low,
            "Whisker_high": self._whisker_high,
            "Array": self.total
        }

        # Создаем JSON-файл в указанной директории
        with open(json_filename, 'w') as json_file:
            json.dump(result, json_file, indent=4)


"""plots = BOXPLOT(path="C:/Users/NightMare/Desktop",
                total=[50, 38, 33, 32, 42, 46, 51, 40, 42, 48])
print(plots)

value = plots.all_value
print(f"Проверка работы метода: {value[0]}")"""


def plot_multiple_boxplots_from_json(directory_path, labels, metrics, titles, float_point):
    """
    Загружает json-файлы из указанной директории, извлекает массивы total и строит боксплоты.

    :param directory_path: Путь к директории с json-файлами.
    :param labels: Список подписей для каждого боксплота.
    """

    # Получаем список файлов в указанной директории
    files = [f for f in os.listdir(directory_path) if f.endswith('.json')]

    # Определяем количество графиков в одной строке (или колонке, в зависимости от предпочтений)
    num_plots = len(files)
    tiks = []

    # Создаем фигуру и одну общую ось для всех графиков
    fig, ax = plt.subplots(figsize=(1.3 * num_plots, 6))

    # Проходимся по каждому файлу и отрисовываем боксплот
    for i, file in enumerate(files):
        file_path = os.path.join(directory_path, file)

        # Загружаем данные из json-файла
        with open(file_path, 'r') as json_file:
            _json = json.load(json_file)

        # Извлекаем массив total
        total = _json.get("array", [])

        # Определяем позицию для текущего боксплота
        pos = i * 0.7 + 1
        tiks.append(pos)

        if i == 0:
            dict_props_mean = {str(i): {"color": "red", "label": "Среднее"} for i in range(num_plots)}
            dict_props_median = {str(i): {"color": "green", "label": "Медиана"} for i in range(num_plots)}
        else:
            dict_props_mean = {str(i): {"color": "red"} for i in range(num_plots)}
            dict_props_median = {str(i): {"color": "green"} for i in range(num_plots)}

        # Строим боксплот на общей оси, уменьшив ширину
        boxplot = ax.boxplot(total,
                             positions=[pos],
                             widths=0.2,
                             meanline=True,
                             showmeans=True,
                             showcaps=True,
                             meanprops=dict_props_mean[f"{i}"],
                             medianprops=dict_props_median[f"{i}"])

        # Получаем значения для надписей из данных JSON
        _median = _json.get("meadian", 0.0)
        _mean = _json.get("mean", 0.0)
        _q1 = _json.get("q1", 0.0)
        _q3 = _json.get("q3", 0.0)
        _whisker_low = _json.get("whisker_low", 0.0)
        _whisker_high = _json.get("whisker_high", 0.0)

        # Добавляем численные значения на график
        ax.text(pos - 0.2, _median, f'{_median:.{float_point}f}', ha='center', va='center', color='green', fontsize=5)
        ax.text(pos + 0.2, _mean, f'{_mean:.{float_point}f}', ha='center', va='center', color='red', fontsize=5)
        ax.text(pos + 0.3, _q1, f'{_q1:.{float_point}f}', ha='center', va='center', color='black', fontsize=5)
        ax.text(pos + 0.3, _q3, f'{_q3:.{float_point}f}', ha='center', va='center', color='black', fontsize=5)
        ax.text(pos + 0.2, _whisker_low, f'{_whisker_low:.{float_point}f}', ha='center', va='center', color='black',
                fontsize=5)
        ax.text(pos + 0.2, _whisker_high, f'{_whisker_high:.{float_point}f}', ha='center', va='center', color='black',
                fontsize=5)

    # Устанавливаем метки для оси X
    ax.set_xticks(list(tiks))
    ax.set_xticklabels(labels)

    ax.set_title(titles)

    # Вывод легенды для медианы и средней линии (применимо к каждому графику)
    ax.legend(loc='lower left')

    # Сохраняем график в отдельный файл
    plt.savefig(os.path.join(directory_path, f"MultipleBoxPlots_{metrics}.png"), dpi=500)

    # Закрываем график
    plt.close()


# Пример использования
"""plot_multiple_boxplots_from_json("C:/Users/NightMare/Desktop/da",
                                 ["Blender", "Automatic", "DreamBooth", "LoRA", "Transformer", "GAN"],
                                 "PSNR",
                                 "Диаграмма размаха метрики пиковое соотношение сигнал/шум (PSNR)",
                                 "2")"""
