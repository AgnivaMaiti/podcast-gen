@echo off
echo Creating Python virtual environment...
python -m venv venv
call venv\Scripts\activate

echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo Installing FFmpeg (required for audio processing)...
pip install ffmpeg-python

echo Setup complete!
echo.
echo To start the application, run:
echo   venv\Scripts\activate

echo   python app.py
