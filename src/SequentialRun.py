#!/usr/bin/env python

# Tasks
# 1) Generate Synthetic Data sets
# 2) Split into train and test
# 3) Add noise
# 4) Cross-validation
# 5) Train model
# 6) Do prediction on test set
# 7) Plot comparison

from TrainingModel import TrainingModel
import pandas as pd
import os

def sequential_run():
    """
Follow the instruction, select type of data
and give noise rate for each class label.
=======================================================>

    """
    cur_path = os.curdir
    os.chdir(cur_path)
    os.chdir("..")
    while True:
        single_run()
        if raw_input("Try more? Y/N ") == 'Y':
            print "\n" * 50
        else:
            break

def single_run(single_call = False):
    if single_call:
        cur_path = os.curdir
        os.chdir(cur_path)
        os.chdir("..")
    data_type = input("What kind of synthetic data do you want to create?\nLinearly(1) or Random(2) or Banana(3)? Please type 1 or 2 or 3: ")

    n = 10000
    po1 = input("Pick a noise rate for class label -1: ")
    po2 = input("Pick a noise rate for class label 1: ")
    run = TrainingModel(n, data_type, po1, po2)
    clf = run.selectClfByKFold(po1, po2)
    run.comparison_plot(clf, po1, po2)

def getbalancedTable(random_data = False):
    rates = [[0.2,0.2],[0.3,0.1],[0.4,0.4]]
    dims  = [9,8,5,29,13]
    ns = [2000,8000,3000,10000,1500]
    table = []
    headers = ["d","n+","n-","p+1","p-1","accuracy"]
    for n,d in zip(ns,dims):
        for r in rates:
            group_run = TrainingModel(n, random_data,r[0], r[1],d)
            clf = group_run.selectClfByKFold(r[0], r[1])
            x_o = [item[1] for item in group_run.noised_test_set]
            y_o = [item[2] for item in group_run.noised_test_set]
            nosiy_test_X1 = zip(x_o, y_o)
            label_o = [group_run.true_data_map[(x, y)] for x, y in nosiy_test_X1]
            pred_label1 = clf.predict(nosiy_test_X1)
            accuracy = group_run.accuracy(label_o, pred_label1)
            table.append((d,group_run.n1,group_run.n2,r[0],r[1],accuracy))
    df = pd.DataFrame(table,columns = headers)
    df.to_csv("src/Tables/TableForSection4.2.1.csv")
    print "Table has been generated!"

def getUnbalancedTable(random_data = False):
    rates = [[0.2, 0.2], [0.3, 0.1], [0.4, 0.4]]
    dims = [9, 8, 5, 29, 13]
    ns = [2000, 8000, 3000, 10000, 1500]
    ws = [[0.3,0.7],[0.4,0.6],[0.25,0.75],[0.45,0.55],[0.6,0.4]]
    table = []
    headers = ["d", "n+", "n-", "p+1", "p-1", "accuracy"]
    for n, d, w in zip(ns, dims, ws):
        for r in rates:
            group_run = TrainingModel(n, random_data, r[0], r[1],d,w)
            clf = group_run.selectClfByKFold(r[0], r[1])
            x_o = [item[1] for item in group_run.noised_test_set]
            y_o = [item[2] for item in group_run.noised_test_set]
            nosiy_test_X1 = zip(x_o,y_o)
            label_o = [group_run.true_data_map[(x, y)] for x, y in nosiy_test_X1]
            pred_label1 = clf.predict(nosiy_test_X1)
            accuracy = group_run.accuracy(label_o, pred_label1)
            table.append((d, group_run.n1, group_run.n2, r[0], r[1], accuracy))
    df = pd.DataFrame(table, columns=headers)
    df.to_csv("src/Tables/TableForSection4.2.2.csv")
    print "Table has been generated!"


if __name__ == "__main__":
    sequential_run()