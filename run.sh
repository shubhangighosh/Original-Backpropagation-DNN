python3 ensemble.py --lr 0.1 --momentum 0.5 --num_hidden 3 --sizes 0 --activation tanh --loss ce --opt nag --batch_size 30 --anneal true --save_dir model/ --expt_dir log/ --train train.csv --val val.csv --test test.csv
python3 train.py --lr 0.1 --momentum 0.5 --num_hidden 3 --sizes 0 --activation tanh --loss ce --opt nag --batch_size 30 --anneal true --save_dir model/ --expt_dir log/ --train train.csv --val val.csv --test test.csv
