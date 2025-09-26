# For single frame:

1) Collect frames
python gesture_singleframe.py collect --flip

2) Train
python gesture_singleframe.py train --index data_single/index_single.csv

3) Live inference
python gesture_singleframe.py infer --model model_mlp_single.pth --flip

# For sequences:

1) Collect frames
python collect_sequences.py

2) Train
python train_tcn.py

3) Live inference
python infer_live.py