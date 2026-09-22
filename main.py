import os
os.environ.setdefault("GLOG_minloglevel", "2")

from core.application import Application

if __name__ == "__main__":
    Application().run()