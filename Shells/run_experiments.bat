@echo Running Reported Experiments
@echo off

python main.py --run_name test_1_1 --input_img assets/119082.jpg --sigma 0.5
python main.py --run_name test_1_2 --input_img assets/119082.jpg --sigma 1.0
python main.py --run_name test_1_3 --input_img assets/119082.jpg --sigma 1.5
python main.py --run_name test_2_1 --input_img assets/58060.jpg --sigma 0.5
python main.py --run_name test_2_2 --input_img assets/58060.jpg --sigma 1.0
python main.py --run_name test_2_3 --input_img assets/58060.jpg --sigma 1.5
python main.py --run_name test_3_1 --input_img assets/42049.jpg --sigma 0.5
python main.py --run_name test_3_2 --input_img assets/42049.jpg --sigma 1.0
python main.py --run_name test_3_3 --input_img assets/42049.jpg --sigma 1.5

echo done!!