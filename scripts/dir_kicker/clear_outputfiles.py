import argparse, pickle, sys, os, json, datetime, pathlib
import pandas as pd

def get_params(filename):
    print(filename)
    ofilepath = os.path.join(f'~/{DIR_KICKER_LOG_PATH}/', filename)
    ofilename = filename
    with open(filename) as inputfile:
        for line in inputfile:
            if 'kfold' in line:
                kfold = int(line.split(' ')[2].strip())
            if 'yymmddhhmmss' in line:
                yymmddhhmmss = line.split(' ')[1].strip()
            if 'checkpoint' in line:
                checkpoint = line.split(' ')[1].strip().replace('/root', '~')
            if 'hh.pickle' in line or 'hh.csv' in line:
                hhpath = '~/' + line.split('/root/')[1].strip()
            if 'yy.pickle' in line or 'yy.csv' in line:
                yypath = '~/' + line.split('/root/')[1].strip()
        
    return kfold, yymmddhhmmss, hhpath, yypath, checkpoint, ofilepath, ofilename


def get_o(dirname='.'):
    files = os.listdir(dirname)
    return [f for f in files if '.sh.o' in f]

def get_e(dirname='.'):
    files = os.listdir(dirname)
    return [f for f in files if '.sh.e' in f]


DIR_KICKER_LOG_PATH = '~/jupyter_notebook/user_work/tcrpred/scripts/dir_kicker/logs'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", default='')
    args = parser.parse_args()
    yymmddhhmmss = datetime.datetime.today().strftime("%Y%m%d_%H%M%S")
    columns = ["knum","yyyymmdd","hhpath","yypath","checkp","sh.o.file","sh.o.name"]
    #assert len(get_o())==5, 'There are more than 5 .sh.o.* files'
    #assert len(get_e())==5, 'There are more than 5 .sh.e.* files'
    df = pd.DataFrame([get_params(f) for f in get_o(dirname='.')], columns=columns)

    outfilename = f'files_{args.j}{yymmddhhmmss}.csv'
    
    """checkpoint sync"""
    os.system('aws s3 sync ~/jupyter_notebook/user_work/tcrpred/scripts/../../checkpoint/ s3://sg-playground-kkoyama-temp/tcrpred/checkpoint/')
    """hhyylog sync"""
    for path in (df.hhpath.tolist() + df.yypath.tolist()):
        command = f'aws s3 cp {path} s3://sg-playground-kkoyama-temp/tcrpred/hhyylog/'
        os.system(command)
    
    
    df.to_csv(f'{DIR_KICKER_LOG_PATH}/{outfilename}', index=None)
    
    for f in get_o(dirname='.'):
        os.system(f'cp {f} ./logs/')
        
    for f in get_e(dirname='.'):
        os.system(f'cp {f} ./logs/')
    
    for f in get_o(dirname='.'):
        os.system(f'rm {f}')

    for f in get_e(dirname='.'):
        os.system(f'rm {f}')
