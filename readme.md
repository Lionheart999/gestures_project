# For single frame:

1) Collect frames
python gesture_singleframe.py collect --flip

2) Train
python gesture_singleframe.py train --index data_single/index_single.csv

3) Live inference
python gesture_singleframe.py infer --model model_mlp_single.pth --flip

# For sequences:

Old sequence method

1) Collect frames
python collect_sequences.py

2) Train
python train_tcn.py

3) Live inference
python infer_live.py

New sequence method

# collect stays the same
python gesture_temporal.py collect --outdir data --seq_len 96 --flip

# train TCN (default)
python gesture_temporal.py train --index data/index.csv --model_type tcn

# train LSTM
python gesture_temporal.py train --index data/index.csv --model_type lstm

# train GRU
python gesture_temporal.py train --index data/index.csv --model_type gru

# train Transformer
python gesture_temporal.py train --index data/index.csv --model_type transformer

# infer with whatever you trained
python gesture_temporal.py infer --model model_tcn.pth --model_type tcn --seq_len 96 --flip
