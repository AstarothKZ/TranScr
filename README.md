# TranScr

Минимальный прототип экранного переводчика для Windows.

Текущий пайплайн:

`Alt+T -> захват всего виртуального рабочего стола -> OCR -> вывод результата в консоль`

## Что сейчас есть

- глобальная горячая клавиша `Alt+T`;
- захват всех доступных мониторов через `mss`;
- сохранение PNG в `test/screenshots` как тестового артефакта;
- OCR в отдельном worker-потоке;
- выбор OCR при запуске:
  - Windows OCR — основной вариант;
  - PaddleOCR 3.7.0 — альтернативный вариант;
- единый формат результата OCR с координатами и confidence;
- `Esc` для завершения;
- простой unit-тест для worker без запуска реального OCR.

## Установка

Используй Python 3.11 на Windows и чистое виртуальное окружение.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Запуск

```powershell
.\.venv\Scripts\python.exe main.py
```

Сначала приложение спросит OCR:

```text
1 - Windows OCR (по умолчанию)
2 - PaddleOCR
```

Затем:

- `Alt+T` — сделать снимок;
- `Esc` — завершить программу.

## Важно про Windows OCR

`Windows.Media.Ocr` использует установленный в Windows OCR-языковой пакет. Microsoft указывает, что этот API поддерживается для desktop-приложений с package identity, поэтому этап упаковки TranScr в MSIX лучше считать обязательной частью перехода от прототипа к полноценному Windows-приложению.

Для проверки альтернативы оставлен PaddleOCR.

## Архитектура

```text
TranScr/
├── core/
│   └── application.py
├── modules/
│   ├── capture/
│   │   └── screen_capture.py
│   └── ocr/
│       ├── base.py
│       ├── ocr_worker.py
│       ├── paddle_ocr.py
│       ├── types.py
│       └── windows_ocr.py
├── test/
│   ├── screenshots/
│   │   └── .gitkeep
│   └── test_ocr_worker.py
├── main.py
├── requirements.txt
└── .gitignore
```

`core` не знает деталей реализации OCR. Любой новый OCR должен реализовать один метод:

```python
def recognize(image: np.ndarray) -> list[OCRText]:
    ...
```

Поэтому следующий модуль можно добавлять отдельно, не переписывая `Application` и worker.

## Следующий этап

После стабилизации этого MVP логично добавлять компоненты по одному:

1. выбор области захвата;
2. отдельный модуль предобработки изображения;
3. обработку текста и фильтры;
4. Google Translator;
5. простой overlay;
6. кэш;
7. настройки и конфигурацию;
8. real-time режим;
9. упаковку и менеджер дополнительных компонентов.
