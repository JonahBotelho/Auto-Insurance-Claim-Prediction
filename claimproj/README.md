# Insurance claim MAE project

Setup (once):
    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    Put trainingset.csv and testset.csv in data/

Run, in this order:
    python cv_harness.py --repeats 3 --models lgbm,cat   # cross-validated MAE (a few minutes); also saves results/oof_*.npy
    python train_final.py --threshold 0.55               # fits on all training rows, writes model.joblib
    python predict.py --output predictions.csv           # predictions for data/testset.csv
    python predict.py --zeros --output zeros.csv         # all-zero baseline
    python gen_submissions.py                            # writes 1_2_1.csv ... 1_2_10.csv (the 10 checkpoint submissions)

For the competition set: python predict.py --input path/to/competition.csv --output out.csv
