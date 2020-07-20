import glob
import pandas as pd
import argparse
import os


def pd2csv(path, result_path, keyword):
    files_prev = glob.glob(path+f'*{keyword}*')
    for fname in files_prev:
        temp = pd.read_pickle(fname).dropna(how="all")
        # some modificatin to our tables
        for key in temp.keys():
            if key in ['Idx', 'Eps', 'prop']:
                if key == 'prop' and ("mnist" in keyword or "cifar10" in keyword):
                    temp = temp.drop(columns=key)
                pass
            elif "SAT" in key:
                temp = temp.rename(columns={key: "SAT"})
            elif "BBran" in key:
                temp = temp.rename(columns={key:"Branches"})
            elif "BTime" in key:
                temp = temp.rename(columns={key:"Time(s)"})
            else:
                temp = temp.drop(columns=key)
        
        fname = fname.split('/')[-1]
        fname_csv = result_path+fname[:-4]+".csv"
        temp.to_csv(fname_csv, index=False)

    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--keyword', type=str, help='modify files whose names containing the keyword')
    parser.add_argument('--path', type=str, default='./cifar_results/', help="path of files to be modified")
    args = parser.parse_args()
    path = args.path
    result_path = path+'csv/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    pd2csv(path, result_path, args.keyword)

