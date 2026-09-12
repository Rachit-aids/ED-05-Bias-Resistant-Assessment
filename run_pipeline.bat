@echo off
python scripts\download_data.py
python scripts\prepare_data.py --raw-dir data\raw --out-dir data\processed
python scripts\train.py --train data\processed\train.csv --dev data\processed\dev.csv --output-dir models\ed05
python scripts\evaluate.py --model-dir models\ed05 --input data\processed\dev.csv --tfidf-train data\processed\train.csv --output-dir outputs\evaluation
