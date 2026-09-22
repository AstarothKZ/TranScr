# TranScr

Минимальный прототип экранного переводчика.

## Что умеет сейчас

- запускаться из `main.py`;
- ждать горячую клавишу `Alt+T`;
- захватывать весь доступный экран;
- сохранять PNG в `test/screenshots`;
- распознавать текст через PP-OCRv6 Small;
- выполнять OCR в отдельном потоке;
- выводить время захвата и OCR в консоль;
- завершаться по `Esc`.

## Структура

```text
TranScr
├── core
│   └── application.py
├── modules
│   ├── capture
│   │   └── screen_capture.py
│   └── ocr
│       ├── paddle_ocr.py
│       └── ocr_worker.py
├── test
│   └── screenshots
├── main.py
├── requirements.txt
└── .gitignore
```

## Версия окружения

Проект рассчитан на:

- Python 3.11+
- PaddleOCR 3.7.0
- PaddlePaddle 3.2.2

Используется PP-OCRv6 Small и CPU-инференс с включённым MKL-DNN.

## Установка

Рекомендуется создавать чистое виртуальное окружение, особенно после смены версии PaddlePaddle.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Проверка окружения:

```powershell
.\.venv\Scripts\python.exe -c "import paddle, paddleocr; print('PaddlePaddle:', paddle.__version__); print('PaddleOCR:', paddleocr.__version__); print('CUDA:', paddle.is_compiled_with_cuda())"
```

Ожидается:

```text
PaddlePaddle: 3.2.2
PaddleOCR: 3.7.0
CUDA: False
```

## Запуск

Из корня проекта:

```powershell
python main.py
```

После запуска:

1. Нажмите `Alt+T`.
2. Скриншот сохранится в `test/screenshots`.
3. Изображение напрямую передастся в OCR из памяти.
4. OCR выполнится в отдельном потоке.
5. Нажмите `Esc` для завершения.

При первом запуске PaddleOCR может загрузить модели в локальный кэш.

## Производительность

PNG сохраняется только как тестовый артефакт. OCR работает с `numpy.ndarray` напрямую.

Внутри PaddleOCR используется до 8 CPU-потоков. Значение рассчитывается автоматически по числу логических процессоров и ограничивается восемью.

На текущем этапе весь экран всё ещё передаётся в OCR. Следующий существенный шаг для скорости — выбор области захвата.

## Важное замечание по PaddlePaddle

Версия `3.3.x` не используется из-за ошибки oneDNN/PIR, которая проявлялась при CPU-инференсе PP-OCRv6. Для текущего прототипа зафиксирована `3.2.2`.
