init.conda:
	conda create -n tcrpred python=3.8.3 -y
	


env.conda:
	conda install -c conda-forge lightgbm -y
	conda install -c plotly plotly=4.14.3 -y
	conda install -y pytorch==1.7.0 cudatoolkit=10.2 -c pytorch
	pip install --upgrade nbformat
	python -m pip install pip==20.2 && pip install -r ./requirements.txt
	python -m ipykernel install --name tcrpred
	sudo python -m ipykernel install --name tcrpred
	conda install -y s3fs

squid.conda:
	conda install -c conda-forge lightgbm -y
	conda install -c plotly plotly=4.14.3 -y
	conda install pytorch torchvision torchaudio cudatoolkit=11.2 -c pytorch -c nvidia
	pip install --upgrade nbformat
	python -m pip install pip==20.2 && pip install -r ./requirements.txt
	python -m ipykernel install --name tcrpred
	sudo python -m ipykernel install --name tcrpred

test2:
	python3 ../main.py --params optuna_best.json --dataset test --kfold 0 --spbtarget LPRRSGAAGA

app:
	streamlit run app.py
