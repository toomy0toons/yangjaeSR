import os
import shutil

source = '/workspace/dacon/HAT/results/HAT-L_SRx4_finetune_submission/visualization/yangjaeSR/'
path = '/workspace/dacon/HAT/submissions/270k'


for f in os.listdir(source):
    os.chdir(source)
    os.rename(f,f[:5]+'.png')

shutil.copytree(source,path,dirs_exist_ok=True)

#zip

shutil.make_archive(path,'zip',path)