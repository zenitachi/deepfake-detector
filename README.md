## Setup

1. Clone the repo
   git clone https://github.com/yourname/deepfake-detector

2. Install dependencies
   pip install -r requirements.txt

3. Download the dataset from Kaggle
   https://kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
   Extract to: D:\DATASETS\deepfake detection\real_vs_fake\

4. Download the model file:
   https://drive.google.com/file/d/1ZkpKo2gSpbwMybHWuabQHAPnnB-4xg87/view?usp=sharing
   
6. Edit the two paths in app.py to match where you extracted the dataset
   (model weights download automatically on first run)

7. Run
   python app.py → open http://127.0.0.1:5000
   
