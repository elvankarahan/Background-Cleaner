import os
import gdown
from zipfile import ZipFile

model_url = "https://drive.google.com/uc?id=1SRT_h-l4D0oVBmtfy4luRyQYAMcEzlFe"

gdown.download(model_url, 'model.zip', quiet= False)

zf = ZipFile('model.zip')
zf.extractall()
os.remove('model.zip')