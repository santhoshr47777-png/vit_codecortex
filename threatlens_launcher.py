import os
import sys

from streamlit.web import bootstrap


def main():
    if getattr(sys, "frozen", False):
        base_dir = sys._MEIPASS
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    app_path = os.path.join(base_dir, "app.py")

    flag_options = {
        "server.headless": False,
        "server.address": "127.0.0.1",
        "server.port": 8501,
        "browser.gatherUsageStats": False,
    }

    bootstrap.run(
        app_path,
        "streamlit run",
        [],
        flag_options
    )


if __name__ == "__main__":
    main()
