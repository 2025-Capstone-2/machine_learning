# machine_learning
1. 추출
``
python stream_video_wifi_detector.py extract -f train_pcap/train_suspicious.pcap -o train_pcap/feat_susp.csv -l train_pcap/lab_susp.dat -w 11 --label 1;python stream_video_wifi_detector.py extract -f train_pcap/train_web.pcap -o train_pcap/feat_web.csv -l train_pcap/lab_web.dat -w 11 --label 0;python stream_video_wifi_detector.py extract -f train_pcap/train_youtube.pcap -o train_pcap/feat_youtube.csv -l train_pcap/lab_youtube.dat -w 11 --label 0
``
``
python stream_video_wifi_detector.py extract -f test_pcap/test_suspicious.pcap -o test_pcap/feat_susp.csv -l test_pcap/lab_susp.dat -w 11 --label 1;python stream_video_wifi_detector.py extract -f test_pcap/test_web.pcap -o test_pcap/feat_web.csv -l test_pcap/lab_web.dat -w 11 --label 0;python stream_video_wifi_detector.py extract -f test_pcap/test_youtube.pcap -o test_pcap/feat_youtube.csv -l test_pcap/lab_youtube.dat -w 11 --label 0
``

2. 학습
``
python stream_video_wifi_detector.py train --features train_pcap/feat_susp.csv train_pcap/feat_youtube.csv train_pcap/feat_web.csv --labels   train_pcap/lab_susp.dat train_pcap/lab_youtube.dat train_pcap/lab_web.dat --epochs 10 --batch 64 -m wifi_dl.pth
``

3. 평가
``
python evaluate_wifi_model.py --features test_pcap/feat_susp.csv test_pcap/feat_youtube.csv test_pcap/feat_web.csv --labels   test_pcap/lab_susp.dat test_pcap/lab_youtube.dat test_pcap/lab_web.dat -m wifi_dl.pth --batch 128
``