# Diffusion music generation

| Название файла/директории | Содержание файла |
|----:|:----------|
| DatasetLoading.ipynb | Ноутбук формата .ipynb в котром качаються данные с текстовым описанием. В нем также есть кастомный dataset из Torch. он не используеться но может понадобиться для обучения другой архитектуры|
| ddpm.py | Файл .py, содержащий все для запуска модели. Файл запускаеться через командную строку и вам нужно укзаать правильное количетсво видеокарт, так как там используеться агреграция градиентовиз обучения на нескольких видеокартах. |
| inferddpm.py  | Файл .py, содержащий все для тестирвоания модели и генерации новых изображений|
| modules.py  | Файл .py с архтиктурой модели для обучения|
| utils.py  | Файл .py с полезными функциями, например для сохранения изображения или его отрисовки|
| audio_transfrs.ipynb | Ноутбук формата .ipynb в котром можно сжать изображения или получить обратно из них музыку|
| splitting.ipynb | Ноутбук формата .ipynb в котром можно нарезать все аудио по 5 секунд|
| wav-image.ipynb | Ноутбук формата .ipynb в котором можно преобразовать аудио в спектограммы|



## Описание
Генерация музыки произведений с помощью диффузионной модели. Цель проекта была в том, чтобы научиться генерировать новую музыку с помощью нейронных сетей. Для обучения использовалась модификация архтиектуры Unet.

Код брался из данной статьи: https://habr.com/ru/companies/ruvds/articles/708182/
Обучение занимает примерно 12 часов на видокарте Tesla для картинок 64 на 64 пикселя. Дальше loss модели уже не падает.

Для запуска нужно прописать в ddpm.py ваши пути и настроить количетсво видеокарт. Для запуска нужно использовать команду: 
python3 -m torch.distributed.run --nproc_per_node=1 ddpm.py

Данный проект делали:
 - Агишев Владимир
 - Теймру Аскеров
 - Влад Подсадный 
