import os
import shutil

source = 'results/HAT-L_Dacon_Submission/visualization/yangjaeSR'
path = 'submissions/dacon_submission'


for f in os.listdir(source):
    os.chdir(source)
    os.rename(f,f[:5]+'.png')

shutil.copytree(source,path,dirs_exist_ok=True)

#zip

shutil.make_archive(path,'zip',path)