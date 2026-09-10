# AI_image_detection
AI Image Detection
A CPU-efficient machine-learning project for classifying images as real or AI-generated under offline, Docker-based runtime constraints.

Overview
The project includes dataset cleaning, duplicate detection, a compact CNN trained with balanced sampling, and a Random Forest baseline. Automatic hyperparameter tuning with Optuna and threshold calibration are used to balance AI-image recall with a controlled false-positive rate.

The final workflow also evaluates robustness to common image transformations, such as JPEG recompression, blur, resizing, and noise, and uses saliency and occlusion analysis to inspect model behaviour.

Technologies
Python, PyTorch, Optuna, scikit-learn, Pillow, NumPy, and Docker.
