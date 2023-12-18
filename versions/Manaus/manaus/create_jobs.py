

import json, os
from rxwgan.datasets import DownloadDataset

token = "b16fe0fc92088c4840a98160f3848839e68b1148"
app = DownloadDataset(token)
app.download('manaus', 'dataset', force_path='/home/brics/public/brics_data/Manaus/manaus/raw')
dataset = os.getcwd()+'/dataset/images.csv'
splits  = os.getcwd()+'/dataset/splits.pic'


output_path = os.getcwd()+'/jobs'
os.makedirs(output_path, exist_ok=True)

tests = 10
sorts = 9

for test in range(tests):
    for sort in range(sorts):

        d = {   
                'sort'   : sort,
                'test'   : test,
                'seed'   : 512,
                'dataset': dataset,
                'splits' : splits,
            }
        print(d)
        o = output_path + '/job.test_%d.sort_%d.json'%(test,sort)
        with open(o, 'w') as f:
            json.dump(d, f)






