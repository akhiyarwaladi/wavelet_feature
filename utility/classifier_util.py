# classifier_util.py
import os
import numpy as np
import json
import itertools
import sklearn.metrics as m
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def analyze(results,outDir):
    print 'analyzing perf ...'
    ##
    perfAllFolds = defaultdict(list)
    for r in results:
        yTrue = r['yTrue']
        yPred = r['yPred']
        xRaw = r['xRaw']

        perfAllFolds['accuracy_score'].append( m.accuracy_score(yTrue,yPred) )
        perfAllFolds['f1_score'].append( m.f1_score(yTrue,yPred,average='micro')  )
        perfAllFolds['precision_score'].append( m.precision_score(yTrue,yPred,average='micro') )
        perfAllFolds['recall_score'].append( m.precision_score(yTrue,yPred,average='micro') )

    ##
    perfMean = dict()
    for k,v in perfAllFolds.iteritems():
        perfMean[k] = np.mean(v)

    ##
    with open(os.path.join(outDir,'perfMean.json'),'w') as f:
        json.dump(perfMean,f,sort_keys=True,indent=2)
    with open(os.path.join(outDir,'perfAllFolds.json'),'w') as f:
        json.dump(perfAllFolds,f,sort_keys=True,indent=2)

    return perfAllFolds

def analyze_performance(estimator_list, outDir):
    f1_score_list = []
    accuracy_score_list = []
    recall_score_list = []
    precision_score_list = []

    bestPerf = estimator_list[0]
    bestPerf_idx = 0
    idx = 0

    for i in estimator_list:
        f1_score_list.append(i['f1_score'])
        accuracy_score_list.append(i['accuracy_score'])
        recall_score_list.append(i['recall_score'])
        precision_score_list.append(i['precision_score'])

        if bestPerf['accuracy_score'] < i['accuracy_score']:
            bestPerf = i
            bestPerf_idx = idx

        idx += 1

    clf_perf = dict(
        n_dataset = len(estimator_list),
        mean_f1_score = np.average(f1_score_list),
        std_f1_score = np.std(f1_score_list),
        mean_accuracy_score = np.average(accuracy_score_list),
        std_accuracy_score = np.std(accuracy_score_list),
        mean_recall_score = np.average(recall_score_list),
        std_recall_score = np.std(recall_score_list),
        mean_precision_score = np.average(precision_score_list),
        std_precision_score = np.std(precision_score_list)
    )
    accuracy_perf = dict(
        accuracy = accuracy_score_list
    )
    # dump section

    # dump to JSON
    with open(os.path.join(outDir,'clf_perf.json'),'w') as f:
        json.dump(clf_perf,f,sort_keys=True,indent=2)

    with open(os.path.join(outDir,'accuracy_perf.json'),'w') as f:
        json.dump(accuracy_perf,f,sort_keys=True,indent=2)

    with open(os.path.join(outDir,'clf_perf_all.json'),'w') as f:
        json.dump(estimator_list,f,sort_keys=True,indent=2)

    # plot
    alpha = 0.5
    label_list = ['f1', 'accuracy', 'recall', 'precision']
    color_list = ['red', 'blue', 'green', 'yellow']
    xIdx = np.arange(len(label_list))
    mean_list = [clf_perf['mean_f1_score'],
                clf_perf['mean_accuracy_score'],
                clf_perf['mean_recall_score'],
                clf_perf['mean_precision_score']]

    std_list = [clf_perf['std_f1_score'],
                clf_perf['std_accuracy_score'],
                clf_perf['std_recall_score'],
                clf_perf['std_precision_score']]

    ax = plt.subplot(111)
    plt.title('Classifier performance (n_dataset = '+str(clf_perf['n_dataset'])+')')
    ax.bar(xIdx, mean_list, align='center', alpha=alpha, color=color_list, yerr=std_list)
    ax.set_xticklabels([' ','f1',' ','accuracy',' ','recall', ' ','precision'])
    ax.set_ylim([0,1])
    ax.set_xlabel('metrics')
    ax.set_ylabel('score performance')

    for i in range(0,len(mean_list)):
        ax.text(xIdx[i], mean_list[i]-0.1, str(round(mean_list[i],3)), horizontalalignment='center', fontsize=15)    
    
    plt.savefig(os.path.join(outDir,'plot_clf_performance.jpg'))
    plt.close()
    plt.close('all')
    plt.gcf().clear()
    '''
    dataset_number = np.arange(1,len(f1_score_list)+1)
    alpha = 0.5

    axF1 = plt.subplot(111)
    plt.title('Classifier f1 Score')
    axF1.bar(dataset_number, f1_score_list, alpha=alpha, color='b', align='center')
    line = axF1.axhline(y=mean_f1_score, c="red",linewidth=2,zorder=0, alpha=alpha, label='mean')
    axF1.set_xlim([0, len(f1_score_list)+1])
    axF1.set_ylim([0,1])
    axF1.set_xticks(dataset_number)
    axF1.set_ylabel('F1 score')
    stdText = plt.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    meanText = plt.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    axF1.legend([stdText, meanText, line], ('std = '+str(round(std_f1_score,5)),
                ('mean = '+str(round(mean_f1_score,5))), 'mean'), framealpha=alpha)

    plt.xlabel('clone dataset number')
    plt.savefig(os.path.join(outDir,'plot_clf_performance_f1.jpg'))
    plt.close()
    plt.close('all')
    plt.gcf().clear()

    axAcc = plt.subplot(111)
    plt.title('Classifier Accuracy Score')
    axAcc.bar(dataset_number, accuracy_score_list, alpha=alpha, color='b', align='center')
    line = axAcc.axhline(y=mean_accuracy_score, c="red", linewidth=2, zorder=0, alpha=alpha, label='mean')
    axAcc.set_xlim([0, len(accuracy_score_list)+1])
    axAcc.set_ylim([0,1])
    axAcc.set_xticks(dataset_number)
    axAcc.set_ylabel('Accuracy score')
    stdText = plt.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    meanText = plt.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    axAcc.legend([stdText, meanText, line], ('std = '+str(round(std_f1_score,5)),
                ('mean = '+str(round(mean_accuracy_score,5))), 'mean'), framealpha=alpha)

    plt.xlabel('clone dataset number')
    plt.savefig(os.path.join(outDir,'plot_clf_performance_acc.jpg'))
    plt.close()
    plt.close('all')
    plt.gcf().clear()

    axRec = plt.subplot(111)
    plt.title('Classifier Recall Score')
    axRec.bar(dataset_number, recall_score_list, alpha=alpha, color='b', align='center')
    line = axRec.axhline(y=mean_recall_score, c="red",linewidth=2,zorder=0, alpha=alpha, label='mean')
    axRec.set_xlim([0, len(recall_score_list)+1])
    axRec.set_ylim([0,1])
    axRec.set_xticks(dataset_number)
    axRec.set_ylabel('recall score')
    stdText = plt.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    meanText = plt.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    axRec.legend([stdText, meanText, line], ('std = '+str(round(std_f1_score,5)),
                ('mean = '+str(round(mean_recall_score,5))), 'mean'), framealpha=alpha)
    
    plt.xlabel('clone dataset number')
    plt.savefig(os.path.join(outDir,'plot_clf_performance_rec.jpg'))
    plt.close()
    plt.close('all')
    plt.gcf().clear()

    axPrec = plt.subplot(111)
    plt.title('Classifier Precision Score')
    axPrec.bar(dataset_number, precision_score_list, alpha=alpha, color='b', align='center')
    line = axPrec.axhline(y=mean_precision_score, c="red",linewidth=2,zorder=0, alpha=alpha, label='mean')
    axPrec.set_xlim([0, len(precision_score_list)+1])
    axPrec.set_ylim([0,1])
    axPrec.set_xticks(dataset_number)
    axPrec.set_ylabel('precision score')
    stdText = plt.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    meanText = plt.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    axPrec.legend([stdText, meanText, line], ('std = '+str(round(std_f1_score,5)),
                ('mean = '+str(round(mean_precision_score,5))), 'mean'), framealpha=alpha)

    plt.xlabel('clone dataset number')
    plt.savefig(os.path.join(outDir,'plot_clf_performance_prec.jpg'))
    plt.close()
    plt.close('all')
    plt.gcf().clear()
    '''
    return bestPerf_idx, bestPerf, clf_perf

def pie_chart_class(y, outDir):
    class_list = np.unique(y)
    class_count = [0] * len(class_list)
    class_counter = dict(zip(class_list, class_count))

    for i in y:
        class_counter[i] += 1
    # dump to JSON
    with open(os.path.join(outDir,'classCount.json'),'w') as f:
        json.dump(class_counter,f,sort_keys=True,indent=2)

    # plot
    label_list = [ k for k in class_counter ]
    value_list = [ v for v in class_counter.values() ]
    explode_list = [0.05 for e in class_counter.values() ]

    total = np.sum(value_list)

    plt.figure(1, figsize=(10,6))
    pie = plt.pie(value_list, shadow=True,explode=explode_list, labels=label_list,  
        autopct = lambda(p): '{:.0f}'.format(p * total / 100), startangle=90)
    plt.axis('equal')
    plt.title('Class pie chart', bbox={'facecolor':'0.8', 'pad':5})
    plt.legend(pie[0], label_list, loc="best")
    plt.savefig(os.path.join(outDir,'pie_class.jpg'))
    plt.close()
    plt.close('all')
    plt.gcf().clear()

def plotPrediction(res,dataDir,outDir):
    foldOutDir = os.path.join(outDir,'plot_prediction')
    os.makedirs(foldOutDir)
    labels = res['labels']
    for j,imgName in enumerate(res['xRaw']):
        yTrue = res['yTrue'][j]
        yPred = res['yPred'][j]
        yPredProb = res['yPredProb'][j]

        # figsize 1300x600
        fig  = plt.figure(figsize=(12, 6))
        grid = gs.GridSpec(1,2,width_ratios=[3, 1])

        # image subplot
        imgPath = os.path.join(dataDir,yTrue,imgName)
        img = plt.imread(imgPath)
        plt.subplot(grid[0])
        plt.imshow(img)
        plt.axis('off')
        plt.title(imgName[:-4]+': '+yTrue.upper())

        # barchart subplot
        ypos = range(len(labels))

        plt.subplot(grid[1])
        barplot = plt.barh(ypos,yPredProb,align='center',alpha=0.75)
        barplot[labels.index(yPred)].set_color('red')
        plt.yticks(ypos,labels)
        plt.xticks(np.arange(0.0,1.1,0.2))
        plt.grid()
        plt.xlabel('probability')

        # save fig to file
        plt.tight_layout()
        plt.savefig(os.path.join(foldOutDir,imgName[:-4]+'_pred.png'))
        plt.close(fig)

def calculate_confusion_matrix(yTrue, yPred):
    classes = [yTrue[0]]

    for c in yTrue:
        if c not in classes:
            classes.append(c)

    class_len = len(classes)

    cm = [[0 for x in range(class_len)] for y in range(class_len)]

    for idx in range(0,len(yTrue)):
        cm[classes.index(yTrue[idx])][classes.index(yPred[idx])] += 1

    return classes, np.array(cm)

def plotConfusionMatrix(yTrue,yPred,outDir, normalize=False,cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    #  confusion matrix
    classes, cm = calculate_confusion_matrix(yTrue, yPred)

    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    normStr = ''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        normStr = '_normalized'

    thresh = np.amax(cm) / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j],3),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.title('Confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(outDir,'conf_mat'+normStr+'.png'))
    plt.close(fig)

def main():
    pass

if __name__ == '__main__':
    main()
