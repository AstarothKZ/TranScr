import os

# Reduce noisy logs from native OCR libraries where supported.
os.environ.setdefault("GLOG_minloglevel", "2")

from core.application import Application


if __name__ == "__main__":
    Application().run()
