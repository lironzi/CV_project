import numpy as np
import ast
import os
import detect


def run(myAnnFileName, buses):
    
    #if weights file in the project folder
    script_dir = os.path.dirname(__file__)
    rel_path = 'best_yolo5_2.pt'
    weights_path = os.path.join(script_dir, rel_path)
    
    
    rel_path=buses
    detect_path=os.path.join(script_dir, rel_path)
    

    #detect(save_img=False,weights='yolov5s.pt', source='data/images',imgsz=512,conf_thres=0.25,iou_thres=0.45)
    
    detect.detect(False,weights_path,detect_path,512,0.25,0.45,myAnnFileName)


    
    
    
    # print ('Running dummy runMe')
    # annFileNameGT = os.path.join(os.getcwd(),'annotationsTrain.txt')
    # writtenAnnsLines = {}
    # annFileEstimations = open(myAnnFileName, 'w+')
    # annFileGT = open(annFileNameGT, 'r')
    # writtenAnnsLines['Ground_Truth'] = (annFileGT.readlines())

    # for k, line_ in enumerate(writtenAnnsLines['Ground_Truth']):

    #     line = line_.replace(' ','')
    #     imName = line.split(':')[0]
    #     anns_ = line[line.index(':') + 1:].replace('\n', '')
    #     anns = ast.literal_eval(anns_)
    #     if (not isinstance(anns, tuple)):
    #         anns = [anns]
    #     corruptAnn = [np.round(np.array(x) + np.random.randint(low = 0, high = 100, size = 5)) for x in anns]
    #     corruptAnn = [x[:4].tolist() + [anns[i][4]] for i,x in enumerate(corruptAnn)]
    #     strToWrite = imName + ':'
    #     if(3 <= k <= 5):
    #         strToWrite += '\n'
    #     else:
    #         for i, ann in enumerate(corruptAnn):
    #             posStr = [str(x) for x in ann]
    #             posStr = ','.join(posStr)
    #             strToWrite += '[' + posStr + ']'
    #             if (i == int(len(anns)) - 1):
    #                 strToWrite += '\n'
    #             else:
    #                 strToWrite += ','
    #     annFileEstimations.write(strToWrite)
