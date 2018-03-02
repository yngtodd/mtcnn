import pytorch

from multi_galsang import CNN
import galsang_utils as utils

from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn

from sklearn.utils import shuffle

import numpy as np
import argparse
import copy
import time

import os
import shutil
from sklearn.metrics import f1_score


parser = argparse.ArgumentParser(description="-----[CNN-classifier]-----")
parser.add_argument("--mode", default="train", help="train: train (with test) a model / test: test saved models")
parser.add_argument("--model", default="non-static", help="available models: rand, static, non-static, multichannel")
parser.add_argument("--dataset", default="TREC", help="available datasets: MR, TREC")
parser.add_argument("--save_model", default="F", help="whether saving model or not (T/F)")
parser.add_argument("--early_stopping", default="F", help="whether to apply early stopping(T/F)")
parser.add_argument("--epoch", default=5, type=int, help="number of max epoch")
parser.add_argument("--learning_rate", default=0.1, type=int, help="learning rate")
parser.add_argument('--num_cvs', type=bool, default=True,
                    help='number of cv_folds to run')
parser.add_argument('--model_name', type=str, default="basic_cnn", help='prob of clf layer dims to dropout')
parser.add_argument('--data_name', type=str, default="big_data_300", help='prob of clf layer dims to dropout')
parser.add_argument('--label_name', type=str, default="labels_subsite_train", help='prob of clf layer dims to dropout')
parser.add_argument('--this_fold', type=str, default=0, help='number of cv')
parser.add_argument('--cuda', type=bool, default=False, help='whether to use cuda')
parser.add_argument('--half_prec', type=bool, default=False, help='whether to use half precision training.')
parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')
parser.add_argument('--data_dir', type=str, default='/Users/youngtodd/data/deidentified_mtcnn/data.pkl', help='where the data lives.')
args = parser.parse_args()


def train(data, params):
    '''
    if params["MODEL"] != "rand":
        # load word2vec
        print("loading word2vec...")
        word_vectors = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

        wv_matrix = []
        for i in range(len(data["vocab"])):
            word = data["idx_to_word"][i]
            if word in word_vectors.vocab:
                wv_matrix.append(word_vectors.word_vec(word))
            else:
                wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))

        # one for UNK and one for zero padding
        wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
        wv_matrix.append(np.zeros(300).astype("float32"))
        wv_matrix = np.array(wv_matrix)
        params["WV_MATRIX"] = wv_matrix
    '''
    params['WV_MATRIX']=data['WV_MATRIX']
    model = CNN(**params)
    if args.cuda:
        model = model.cuda()
    #model.share_memory()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])
    criterion = nn.CrossEntropyLoss()

    pre_dev_acc = 0
    max_test_acc = 0
    for e in range(params["EPOCH"]):
        etime = time.time()
        #data["train_x"], data["train_y"] = shuffle(data["train_x"], data["train_y"])
        for i in range(0, len(data["train_x"]), params["BATCH_SIZE"]):
            batch_range = min(params["BATCH_SIZE"], len(data["train_x"]) - i)

            batch_x = [sent.tolist() for sent in data["train_x"][i:i + batch_range]]
            batch_y0 = [data["class0"].index(c) for c in data["train_y0"][i:i + batch_range]]
            batch_y1 = [data["class1"].index(c) for c in data["train_y1"][i:i + batch_range]]
            batch_y2 = [data["class2"].index(c) for c in data["train_y2"][i:i + batch_range]]
            batch_y3 = [data["class3"].index(c) for c in data["train_y3"][i:i + batch_range]]

            batch_x = Variable(torch.LongTensor(batch_x))
            batch_y0 = Variable(torch.LongTensor(batch_y0))
            batch_y1 = Variable(torch.LongTensor(batch_y1))
            batch_y2 = Variable(torch.LongTensor(batch_y2))
            batch_y3 = Variable(torch.LongTensor(batch_y3))
            if args.cuda:
                batch_x = batch_x.cuda()
                batch_y0 = batch_y0.cuda()
                batch_y1 = batch_y1.cuda()
                batch_y2 = batch_y2.cuda()
                batch_y3 = batch_y3.cuda()
                if args.half_prec:
                    batch_x = batch_x.half()
                    batch_y0 = batch_y0.half()
                    batch_y1 = batch_y1.half()
                    batch_y2 = batch_y2.half()
                    batch_y3 = batch_y3.half()
            optimizer.zero_grad()
            model.train()
            pred_task0, pred_task1, pred_task2, pred_task3 = model(batch_x)
            loss_task0 = criterion(pred_task0, batch_y0)
            loss_task1 = criterion(pred_task1, batch_y1)
            loss_task2 = criterion(pred_task2, batch_y2)
            loss_task3 = criterion(pred_task3, batch_y3)
            loss = loss_task0 + loss_task1 + loss_task2 + loss_task3
            loss.backward()
            optimizer.step()

            # constrain l2-norms of the weight vectors
            '''
            if model.fc.weight.norm() > params["NORM_LIMIT"]:
                model.fc.weight.data = model.fc.weight.data * params["NORM_LIMIT"] / model.fc.weight.data.norm()
            '''
        print('epoch training finished in',time.time()-etime)
        #dev_acc = test(data, model, params, mode="dev")
        test_acc = test(data, model, params)
        #print("epoch:", e + 1, "/ dev_acc:", dev_acc, "/ test_acc:", test_acc, "time", time.time()-etime)
        print("epoch:", e + 1, test_acc, "time", time.time()-etime)

    #    if params["EARLY_STOPPING"] and dev_acc <= pre_dev_acc:
    #        print("early stopping by dev_acc!")
    #        break
    #    else:
    #        pre_dev_acc = dev_acc

    #    if test_acc > max_test_acc:
    #        max_test_acc = test_acc
    #        best_model = copy.deepcopy(model)

    #print("max test acc:", max_test_acc)
    #return best_model


def test(data, model, params, mode="test"):
    model.eval()

    if mode == "dev":
        x, y = data["dev_x"], data["dev_y"]
    elif mode == "test":
        test_x = data["test_x"]
        y0 = data["test_y0"]
        y1 = data["test_y1"]
        y2 = data["test_y2"]
        y3 = data["test_y3"]
    '''
    x = [[data["word_to_idx"][w] if w in data["vocab"] else params["VOCAB_SIZE"] for w in sent] +
         [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
         for sent in x]
    '''
    #x = Variable(torch.LongTensor(x))
    all_x0=[]; all_x1=[]
    all_x2=[]; all_x3=[]
    for this_x in test_x:
        this_x = Variable(torch.LongTensor(this_x))
        if args.cuda:
            this_x = this_x.cuda()
        pred_task0, pred_task1, pred_task2, pred_task3 = model(this_x)

        pred_task0 = np.argmax(pred_task0.cpu().data.numpy(), axis=1)
        pred_task1 = np.argmax(pred_task1.cpu().data.numpy(), axis=1)
        pred_task2 = np.argmax(pred_task2.cpu().data.numpy(), axis=1)
        pred_task3 = np.argmax(pred_task3.cpu().data.numpy(), axis=1)

        all_x0.extend(pred_task0)
        all_x1.extend(pred_task1)
        all_x2.extend(pred_task2)
        all_x3.extend(pred_task3)

    y0 = [data["class0"].index(c) for c in y0]
    y0 = [data["class1"].index(c) for c in y1]
    y0 = [data["class2"].index(c) for c in y2]
    y0 = [data["class3"].index(c) for c in y3]

    acc_task0 = sum([1 if p == y else 0 for p, y in zip(all_x0, y0)]) / len(y0)
    acc_task1 = sum([1 if p == y else 0 for p, y in zip(all_x1, y1)]) / len(y1)
    acc_task2 = sum([1 if p == y else 0 for p, y in zip(all_x2, y2)]) / len(y2)
    acc_task3 = sum([1 if p == y else 0 for p, y in zip(all_x3, y3)]) / len(y3)

    #micro,macro=score(all_x,y)
    print("accuracy for subsite ",acc_task0)
    print("accuracy for laterality ",acc_task1)
    print("accuracy for behavior ",acc_task2)
    print("accuracy for histological grade: ",acc_task3)
    #print("micro",micro)
    #print("macro",macro)
    #return acc_task0


def main():
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)


    data = {}
    data[ 'WV_MATRIX' ] = wv_mat
    data[ 'train_x' ] = train_x
    data[ 'train_y0' ] = train_y0
    data[ 'train_y1' ] = train_y1
    data[ 'train_y2' ] = train_y2
    data[ 'train_y3' ] = train_y3
    data[ 'test_x' ] = test_x
    data[ 'test_y0' ] = test_y0
    data[ 'test_y1' ] = test_y1
    data[ 'test_y2' ] = test_y2
    data[ 'test_y3' ] = test_y3
    data[ 'CLASS_ZERO_SIZE' ] = num_classes[0]
    data[ 'CLASS_ONE_SIZE' ] = num_classes[1]
    data[ 'CLASS_TWO_SIZE' ] = num_classes[2]
    data[ 'CLASS_THREE_SIZE' ] = num_classes[3]

    data["vocab"] = sorted(list(set([w for sent in data["train_x"] + data["test_x"] for w in sent])))
    data["class0"] = sorted(list(set(data["train_y0"])))
    data["class1"] = sorted(list(set(data["train_y1"])))
    data["class2"] = sorted(list(set(data["train_y2"])))
    data["class3"] = sorted(list(set(data["train_y3"])))
    data["word_to_idx"] = {w: i for i, w in enumerate(data["vocab"])}
    data["idx_to_word"] = {i: w for i, w in enumerate(data["vocab"])}

    params = {
        "MODEL": args.model,
        "DATASET": args.dataset,
        "SAVE_MODEL": bool(args.save_model == "T"),
        "EARLY_STOPPING": bool(args.early_stopping == "T"),
        "EPOCH": args.epoch,
        "LEARNING_RATE": args.learning_rate,
        "MAX_SENT_LEN": max([len(sent) for sent in data["train_x"] + data["test_x"]]),
        "BATCH_SIZE": 16,
        "WORD_DIM": 300,
        "VOCAB_SIZE": data['WV_MATRIX'].shape[0]-2,
        #"CLASS_SIZE": len(data["classes"]), # potentially unused?
        "FILTERS": [3, 4, 5],
        "FILTER_NUM": [100, 100, 100],
        "DROPOUT_PROB": 0.5,
        "NORM_LIMIT": 0.001,
        "CLASS_ZERO_SIZE": num_classes[0],
        "CLASS_ONE_SIZE": num_classes[1],
        "CLASS_TWO_SIZE": num_classes[2],
        "CLASS_THREE_SIZE": num_classes[3]
    }

    print("=" * 20 + "INFORMATION" + "=" * 20)
    print("MODEL:", params["MODEL"])
    print("MAX_SENT_LEN:", params["MAX_SENT_LEN"])
    print("VOCAB_SIZE:", params["VOCAB_SIZE"])
    print("EPOCH:", params["EPOCH"])
    print("LEARNING_RATE:", params["LEARNING_RATE"])
    print("EARLY_STOPPING:", params["EARLY_STOPPING"])
    print("SAVE_MODEL:", params["SAVE_MODEL"])
    print("=" * 20 + "INFORMATION" + "=" * 20)
    print("WV_SHAPE", data['WV_MATRIX'].shape)

    if args.mode == "train":
        print("=" * 20 + "TRAINING STARTED" + "=" * 20)
        model = train(data, params)
        if params["SAVE_MODEL"]:
            utils.save_model(model, params)
        print("=" * 20 + "TRAINING FINISHED" + "=" * 20)
    else:
        model = utils.load_model(params)
        if args.cuda:
            model = model.cuda()
        #model = utils.load_model(params)
        test_acc = test(data, model, params)
        print("test acc:", test_acc)


def score(actual,predicted):
    micro_f = f1_score(actual,predicted,average = 'micro')
    macro_f = f1_score(actual,predicted,average = 'macro')
    print("micro-f", micro_f)
    print("macro-f", macro_f)
    return

if __name__ == "__main__":
    main()
