train:
	source env.sh
	python -m ham_radio.train_ls16 with cnn
eval:
	source env.sh
	python -m ham_radio.evaluation with model_dir=/net/vol/werning/ham_radio/models/ham_radio/{model_name}/ num_jobs=16
	python eval_summary.py --model_name={model_name}
