state:
	qstate

mcpas.kfold:
	bash kicker_mcpaskfold.sh
    
mcpas.single:
	qsub  -v "kfold=0" kicker_mcpas.sh

clean:
	python3 clear_outputfiles.py 
    
vdjdbno10x.single:
	qsub  -v "kfold=0" kicker_vdjdbno10x.sh

vdjdbno10x.kfold:
	qsub  -v "kfold=0" kicker_vdjdbno10x.sh && sleep 5
	qsub  -v "kfold=1" kicker_vdjdbno10x.sh && sleep 5
	qsub  -v "kfold=2" kicker_vdjdbno10x.sh && sleep 5
	qsub  -v "kfold=3" kicker_vdjdbno10x.sh && sleep 5
	qsub  -v "kfold=4" kicker_vdjdbno10x.sh && sleep 5

alltrain.kfold:
	qsub  -v "kfold=0,dataset=alltrain,spbtarget=None" kicker.sh && sleep 5
	qsub  -v "kfold=1,dataset=alltrain,spbtarget=None" kicker.sh && sleep 5
	qsub  -v "kfold=2,dataset=alltrain,spbtarget=None" kicker.sh && sleep 5
	qsub  -v "kfold=3,dataset=alltrain,spbtarget=None" kicker.sh && sleep 5
	qsub  -v "kfold=4,dataset=alltrain,spbtarget=None" kicker.sh && sleep 5

all.kfold:
	qsub  -v "kfold=0,dataset=all,spbtarget=None" kicker.sh && sleep 5
	qsub  -v "kfold=1,dataset=all,spbtarget=None" kicker.sh && sleep 5
	qsub  -v "kfold=2,dataset=all,spbtarget=None" kicker.sh && sleep 5
	qsub  -v "kfold=3,dataset=all,spbtarget=None" kicker.sh && sleep 5
	qsub  -v "kfold=4,dataset=all,spbtarget=None" kicker.sh && sleep 5

alltrain.kfold.single:
	qsub  -v "kfold=0,dataset=alltrain,spbtarget=LPRRSGAAGA" kicker.sh && sleep 5
	qsub  -v "kfold=1,dataset=alltrain,spbtarget=LPRRSGAAGA" kicker.sh && sleep 5
	qsub  -v "kfold=2,dataset=alltrain,spbtarget=LPRRSGAAGA" kicker.sh && sleep 5
	qsub  -v "kfold=3,dataset=alltrain,spbtarget=LPRRSGAAGA" kicker.sh && sleep 5
	qsub  -v "kfold=4,dataset=alltrain,spbtarget=LPRRSGAAGA" kicker.sh && sleep 5


spb.single:
	qsub  -v "kfold=0" kicker_vdjdbno10x.sh LPRRSGAAGA
    
spb.5folds.LPRRSGAAGA:
	qsub  -v "kfold=0,spbtarget=LPRRSGAAGA" kicker_mcpas.sh  && sleep 5
	qsub  -v "kfold=1,spbtarget=LPRRSGAAGA" kicker_mcpas.sh  && sleep 5
	qsub  -v "kfold=2,spbtarget=LPRRSGAAGA" kicker_mcpas.sh  && sleep 5
	qsub  -v "kfold=3,spbtarget=LPRRSGAAGA" kicker_mcpas.sh  && sleep 5
	qsub  -v "kfold=4,spbtarget=LPRRSGAAGA" kicker_mcpas.sh  && sleep 5

fakedata:
	qsub -v "kfold=4,spbtarget=None,dataset=fake_v0" kicker.sh
    
sync.squid2s3:
	aws s3 sync ~/jupyter_notebook/user_work/hhyylog/ s3://sg-playground-kkoyama-temp/tcrpred/hhyylog/
	aws s3 sync /sqfs/work/K2107/u6b233/hhyylog/ s3://sg-playground-kkoyama-temp/tcrpred/hhyylog/
	aws s3 sync ~/jupyter_notebook/user_work/checkpoint/ s3://sg-playground-kkoyama-temp/tcrpred/checkpoint/
	aws s3 sync /sqfs/work/K2107/u6b233/checkpoint/ s3://sg-playground-kkoyama-temp/tcrpred/checkpoint/
    
sync.s32mac:
	aws s3 sync  s3://sg-playground-kkoyama-temp/tcrpred/hhyylog/ ~/workspace/hhyylog/
	aws s3 sync  s3://sg-playground-kkoyama-temp/tcrpred/checkpoint/ ~/workspace/checkpoint/

test:
	qsub  -v "kfold=0,dataset=alltrain,spbtarget=LPRRSGAAGA" kicker.sh 

spbtest:
	qsub  -v "kfold=0,spbtarget=LPRRSGAAGA" check_spbtest.sh

