# Deepfake Detector Website

A web interface for detecting real and deepfake videos using a trained PyTorch model.

## What this project includes

- `frontend.py`: Flask-based website for uploading videos and viewing predictions.
- `templates/`: HTML templates for the website pages.
- `static/`: Static assets and upload storage.
- `models/deepfake_detector_best.pt`: Trained checkpoint used by the detector.
- `app.py`: Optional Streamlit demo application.
- `train_deepfake_detector.py`: Model and training utilities.
- `predict_deepfake.py`: CLI script for offline prediction.

## Features

- Upload video files (MP4, MOV, AVI, WebM, MKV)
- Detect whether a video is real or fake
- Display prediction confidence and probabilities
- Website navigation with About / Guide / API pages
- Local deployment and cloud-ready instructions

## Requirements

- Python 3.10+
- `Flask`
- `torch`
- `torchvision`
- `opencv-python`
- `Pillow`
- `numpy`
- `streamlit` (optional, only for `app.py`)

## Installation

1. Clone or download the repository.
2. Open a terminal in the project root:

   ```bash
   cd /Users/roshanrathod/Downloads/DeepfakeDetector-main
   ```

3. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Run the website locally

1. Start the Flask website:

   ```bash
   python3 frontend.py
   ```

2. Open the website in your browser:

   - `http://127.0.0.1:8501`

3. If port `8501` is already in use, run on a different port:

   ```bash
   PORT=8502 python3 frontend.py
   ```

## Optional: Run the Streamlit demo

If you want the Streamlit version instead of the Flask website, use:

```bash
streamlit run app.py
```

Then open:

- `http://localhost:8502`

## Deployment

### Recommended production server

Use `gunicorn` to serve the Flask app in production.

```bash
pip install gunicorn
PORT=8000 gunicorn frontend:app --workers 4 --bind 0.0.0.0:8000
```

### Deploy on Render / Railway / Heroku

1. Push this repository to GitHub.
2. Create a new app on Render, Railway, or Heroku.
3. Set the start command to:

   ```bash
   gunicorn frontend:app
   ```

4. Configure the environment variable `PORT` if required by the host.
5. Make sure the model file `models/deepfake_detector_best.pt` is included in the repository.

### Deployment notes

- The website expects the model checkpoint at `models/deepfake_detector_best.pt`.
- `static/uploads/` is used for temporary upload storage.
- For production, use a file cleanup policy or temporary storage mechanism.
- On cloud hosts, ensure the instance has enough CPU/memory for `torch` and `opencv`.

## Make the website public

To make this site publicly accessible, deploy it on a hosting service like Render, Railway, or Heroku. Once the app is deployed, anyone can browse to the app URL and use the deepfake detector.

If you want the site to be searchable by others, use a public domain or subdomain and make sure the app is not protected by authentication. Search engines can index the site automatically once it is live.

### Recommended steps for public deployment

1. Push your repo to GitHub.
2. Connect the GitHub repo to a hosting provider (Render, Railway, or Heroku).
3. Use `Procfile` to tell the host how to start the app.
4. The host will use `requirements.txt` and `runtime.txt` to install dependencies and select the Python version.
5. Visit the public URL provided by the host.

### Example: deploy on Heroku

```bash
heroku create your-app-name
git push heroku main
heroku ps:scale web=1
```

### Example: deploy on Render

- Create a new web service.
- Point it to the GitHub repo.
- Set the build command to `pip install -r requirements.txt`.
- Set the start command to `gunicorn frontend:app`.

### Example: deploy on Railway

- Create a new project.
- Link your GitHub repo.
- Set the start command to `gunicorn frontend:app`.
- Make sure the `PORT` environment variable is supported.

## Project structure

```text
DeepfakeDetector-main/
├── app.py
├── frontend.py
├── predict_deepfake.py
├── train_deepfake_detector.py
├── models/
│   ├── deepfake_detector_best.pt
│   └── training_metrics.json
├── static/
│   └── uploads/
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── about.html
│   ├── guide.html
│   └── api.html
└── README.md
```

## Troubleshooting

- If the server fails because `port 8501 is in use`, change the port using the `PORT` environment variable.
- If the model fails to load, confirm that `models/deepfake_detector_best.pt` exists.
- If dependencies are missing, reinstall with:

  ```bash
  pip install -r requirements.txt
  ```

## License

This project is provided as-is for demonstration and deployment of a deepfake detection website.
