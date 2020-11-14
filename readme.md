# Reimplementation of 6D Dynamic Camera Relocalization from Single Reference Image

## Project Setup

To set up this project's virtual environment, navigate to the root directory (the one this file is in) and run `python3 -m venv venv` to create a virtual environment. Then, do `source venv/bin/activate` on MacOS/Linux or run `venv\Scripts\activate` on Windows to enter this virtual environment. Finally, do `pip3 install -r requirements.txt` to install this project's dependencies. Make sure you're in the project's virtual environment when trying to run any of the project files.

The recommended way to edit and run this project is through VS Code. If you're using VS Code, you can use launch configurations in `.vscode/launch.json` to run common tasks, and linting will happen automatically. If not, you should run scripts as modules from the root directory (e.g. to run `camera_render.py`, you would do `python3 -m source.scripts.camera_render`).
