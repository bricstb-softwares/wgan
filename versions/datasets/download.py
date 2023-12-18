
import json, os
from rxwgan.datasets import DownloadDataset

token = "b16fe0fc92088c4840a98160f3848839e68b1148"
app = DownloadDataset(token)

basepath = '/home/brics/public/brics_data/Manaus/manaus/raw'
app.download('manaus', 'manaus', force_path=basepath)

basepath = '/home/brics/public/brics_data/Manaus/c_manaus/raw'
app.download('c_manaus', 'c_manaus', force_path=basepath)

basepath = '/home/brics/public/brics_data/Shenzhen/raw'
app.download('china', 'china', force_path=basepath)

basepath = '/home/brics/public/brics_data/SantaCasa/imageamento_anonimizado_valid/raw'
app.download('imageamento_anonimizado_valid', 'imageamento_anonimizado_valid', force_path=basepath)
