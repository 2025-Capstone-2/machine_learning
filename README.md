# machine_learning
1. 추출
## 일반용
``
python stream_video_wifi_detector.py extract -f train_pcap/train_suspicious.pcap -o train_pcap/feat_susp.csv -l train_pcap/lab_susp.dat -w 11 --label 1;python stream_video_wifi_detector.py extract -f train_pcap/train_web.pcap -o train_pcap/feat_web.csv -l train_pcap/lab_web.dat -w 11 --label 0;python stream_video_wifi_detector.py extract -f train_pcap/train_youtube.pcap -o train_pcap/feat_youtube.csv -l train_pcap/lab_youtube.dat -w 11 --label 0
``
``
python stream_video_wifi_detector.py extract -f test_pcap/test_suspicious.pcap -o test_pcap/feat_susp.csv -l test_pcap/lab_susp.dat -w 11 --label 1;python stream_video_wifi_detector.py extract -f test_pcap/test_web.pcap -o test_pcap/feat_web.csv -l test_pcap/lab_web.dat -w 11 --label 0;python stream_video_wifi_detector.py extract -f test_pcap/test_youtube.pcap -o test_pcap/feat_youtube.csv -l test_pcap/lab_youtube.dat -w 11 --label 0
``
## LSTM용
``
python stream_video_wifi_detector.py extract --pcap train_pcap/train_suspicious.pcap --output train_pcap/feat_susp.csv --labels train_pcap/lab_susp.dat --window 11 --label 1 --model-type lstm;python stream_video_wifi_detector.py extract --pcap train_pcap/train_web.pcap --output train_pcap/feat_web.csv --labels train_pcap/lab_web.dat --window 11 --label 0 --model-type lstm;python stream_video_wifi_detector.py extract --pcap train_pcap/train_youtube.pcap --output train_pcap/feat_youtube.csv --labels train_pcap/lab_youtube.dat --window 11 --label 0 --model-type lstm
``
``
python stream_video_wifi_detector.py extract --pcap test_pcap/test_suspicious.pcap --output test_pcap/feat_susp.csv --labels test_pcap/lab_susp.dat --window 11 --label 1 --model-type lstm;python stream_video_wifi_detector.py extract --pcap test_pcap/test_web.pcap --output test_pcap/feat_web.csv --labels test_pcap/lab_web.dat --window 11 --label 0 --model-type lstm;python stream_video_wifi_detector.py extract --pcap test_pcap/test_youtube.pcap --output test_pcap/feat_youtube.csv --labels test_pcap/lab_youtube.dat --window 11 --label 0 --model-type lstm
``
## Anomaly용
``
python stream_video_wifi_detector.py anomaly --features train_pcap/feat_youtube.csv train_pcap/feat_web.csv -m anomaly_ocsvm.pkl --nu 0.05
``
2. 학습
## 일반용
``
python stream_video_wifi_detector.py train --features train_pcap/feat_susp.csv train_pcap/feat_youtube.csv train_pcap/feat_web.csv --labels   train_pcap/lab_susp.dat train_pcap/lab_youtube.dat train_pcap/lab_web.dat --epochs 10 --batch 64 -m wifi_dl.pth
``
## LSTM용
``
python stream_video_wifi_detector.py train --features train_pcap/feat_susp.csv train_pcap/feat_youtube.csv train_pcap/feat_web.csv --labels   train_pcap/lab_susp.dat train_pcap/lab_youtube.dat train_pcap/lab_web.dat --model-type lstm --epochs 20 --batch 128 -m wifi_lstm.pth
``

3. 평가
## 일반용
``
python evaluate_wifi_model.py --features test_pcap/feat_susp.csv test_pcap/feat_youtube.csv test_pcap/feat_web.csv --labels   test_pcap/lab_susp.dat test_pcap/lab_youtube.dat test_pcap/lab_web.dat -m wifi_dl.pth --batch 128
``
## LSTM용
``
python evaluate_wifi_model.py --features test_pcap/feat_susp.csv test_pcap/feat_youtube.csv test_pcap/feat_web.csv --labels   test_pcap/lab_susp.dat  test_pcap/lab_youtube.dat  test_pcap/lab_web.dat --window 11 --model-type lstm -m wifi_lstm.pth --batch 128
``


<!-- python stream_video_wifi_detector.py anomaly --features feat_youtube.csv feat_web.csv -m anomaly_ocsvm.pkl --nu 0.01


python stream_video_wifi_detector.py train --features auto_labeled_X.csv --labels auto_labeled_y.dat --window 11 --model-type lstm -m wifi_lstm_auto.pth --epochs 20

type feat_susp.csv feat_web.csv feat_youtube.csv > test_features.csv
type test_pcap/lab_susp.dat test_pcap/lab_web.dat test_pcap/lab_youtube.dat > test_pcap/lab_features.dat

Get-Content -Path feat_susp.csv,feat_web.csv,feat_youtube.csv |
  Set-Content test_features.csv
Get-Content -Path lab_susp.dat,lab_web.dat,lab_youtube.dat |
  Set-Content test_features.dat
  
python evaluate_wifi_model.py --features test_pcap/test_features.csv --labels test_pcap/test_labels.dat --window 11 --model-type lstm -m wifi_lstm_auto.pth --batch 128 -->

python extract_flow_features.py ` --pcaps train_pcap/train_suspicious.pcap train_pcap/train_youtube.pcap train_pcap/train_web.pcap ` --labels 1 0 0 ` --output train_pcap/flow_feats.csv ` --labelfile train_pcap/flow_labels.dat

python stream_video_wifi_detector.py train --features train_pcap/flow_feats.csv --labels train_pcap/flow_labels.dat --model-type ffnn -m flow_ffnn.pth --epochs 20 --batch 128

python extract_flow_features.py --pcaps train_pcap/train_suspicious.pcap train_pcap/train_youtube.pcap train_pcap/train_web.pcap --labels 1 0 0 -o train_pcap/flow_feats.csv -l train_pcap/flow_labels.dat

python extract_flow_features.py --pcaps test_pcap/test_suspicious.pcap test_pcap/test_youtube.pcap test_pcap/test_web.pcap --labels 1 0 0 -o test_pcap/flow_feats_test.csv -l test_pcap/flow_labels_test.dat

python stream_video_wifi_detector.py train --features train_pcap/flow_feats.csv --labels train_pcap/flow_labels.dat --model-type ffnn -m best_flow_ffnn.pth --epochs 30 --batch 64

python evaluate_wifi_model.py --features test_pcap/flow_feats_test.csv --labels test_pcap/flow_labels_test.dat --window 1 --model-type ffnn -m best_flow_ffnn.pth --batch 64