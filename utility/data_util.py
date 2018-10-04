# data_util.py
import os
import datetime
import socket
import numpy as np

idn2eng = {'jamur':'fungi','bakteri':'bacteria','tidakdiketahui':'unknown'}

def clean(s):
  s = s.strip().replace('\n','').replace('\t','')
  s = ' '.join(s.split())
  return s

def load(dirpath):
  x = []; y = []
  classList = os.listdir(dirpath)
  for c in classList:
    cDir = os.path.join(dirpath,c)
    imgFnames = [f for f in os.listdir(cDir)
                 if os.path.isfile(os.path.join(cDir,f))
                 and (f.endswith(".jpg") or f.endswith('.JPG'))]

    for f in imgFnames:
      x.append(f)
      y.append(c)

  return (x,y)

def tag():
  timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  hostname = socket.gethostname()
  return hostname+'-'+timestamp

def pie_chart_class(y, outDir):
    class_list = np.unique(y)
    class_count = [0] * len(class_list)
    class_counter = dict(zip(class_list, class_count))

    for i in y:
        class_counter[i] += 1
