import time
import argparse
import os
import gc
import random
import math
import numpy as np
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from load_data import *
from encoder import *
from decoder import *
from utils import *

import logging
from torch.utils.tensorboard import SummaryWriter

class Experiment:
    def __init__(self, args):
        print("Experiment init")
        self.args = args

        self.cached_sample = {}
        self.best_result = ()
        self.early_stop_val_result = ()

    def init_embeddings(self):
        print("init_embeddings")
        # 创建实体和关系嵌入向量矩阵，并将其移动到设备上
        self.rel_embeddings = nn.Embedding(d.rel_num, self.args.hiddens[0]).to(device)
        nn.init.xavier_normal_(self.rel_embeddings.weight)
        if self.args.ent_init == "random":
            self.ent_embeddings = nn.Embedding(d.ent_num, self.args.hiddens[0]).to(device)
            nn.init.xavier_normal_(self.ent_embeddings.weight)
        elif self.args.ent_init == "name":
            with open(file= self.args.data_dir + '/vectorList.json', mode='r', encoding='utf-8') as f:
                embedding_list = json.load(f)
                print(len(embedding_list), 'rows,', len(embedding_list[0]), 'columns.')
                input_embeddings = torch.tensor(embedding_list)  # 将 embedding_list 转换为 PyTorch 张量
                self.ent_embeddings = nn.Embedding(
                    num_embeddings=input_embeddings.shape[0],  # 嵌入矩阵的行数
                    embedding_dim=input_embeddings.shape[1]  # 嵌入向量的维度
                )
                self.ent_embeddings.weight.data.copy_(input_embeddings)
        else:
            raise NotImplementedError("bad ent_init")
        
        # 初始化 encoder 之后的 enh_ent_embeddings 为 numpy 数组
        self.enh_ent_embeddings = self.ent_embeddings.weight.cpu().detach().numpy()

    def train_and_eval(self):
        print("train_and_eval")
        self.init_embeddings()
        print("self.ent_embeddings.weight", self.ent_embeddings.weight)
        # 初始化 encoder 和 decoder
        encoder = Encoder(name=self.args.encoder, hiddens=self.args.hiddens, heads=self.args.heads+[1], skip_conn=self.args.skip_conn,activation=self.args.activation, feat_drop=self.args.feat_drop, bias=self.args.bias, ent_num = d.ent_num, rel_num = d.rel_num).to(device)
        logger.info(encoder)  # 输出encoder信息

        decoder = Decoder(name=self.args.decoder, hiddens=self.args.hiddens, skip_conn=None, train_dist=self.args.train_dist, sampling=self.args.sampling, alpha=self.args.alpha, margin=self.args.margin).to(device)
        logger.info(decoder)
        
        # 定义参数列表和优化器
        params = nn.ParameterList([self.ent_embeddings.weight, self.rel_embeddings.weight] + (list(decoder.parameters())) + (list(encoder.parameters())) )
        opt = optim.Adagrad(params, lr=self.args.lr)
        logger.info(params)
        logger.info(opt)
        

        # 传入模型所需额外信息
        others = None
        # Dual-AMN
        if encoder.name == "dual-amn":
            others = [d.triple_num, d.ent_adj, d.rel_adj, d.r_val, d.adj_input, d.r_index]


        # 训练
        logger.info("Start training...")
        patience_cnt = 0
        for it in range(0, self.args.epoch): # 循环 epoch 次 

            if (len(d.ill_train_idx) == 0):  # 如果没有训练数据，则跳过该解码器的训练
                continue

            t_ = time.time()  # 记录开始时间

            # 训练一个 epoch,得到 loss,同时得到 self.enh_ent_embbeddings,即加强后的实体嵌入
            loss = self.train_1_epoch(it, opt, encoder, decoder, d.sparse_edges_idx, d.sparse_values, d.sparse_rels_idx, d.triple_idx, d.ill_train_idx, [d.kg1_ent_ids, d.kg2_ent_ids], self.ent_embeddings.weight, self.rel_embeddings.weight, others)
            writer.add_scalar("loss", loss, it)  # 在tensorboard中记录损失值
            logger.info("epoch %d: %.8f\ttime: %ds" % (it, loss, int(time.time()-t_)) )  # 输出当前迭代的训练结果

            # 先每一个 epoch val 一下
            logger.info("Start validating on val set if val_rate > 0 else test set:")
            with torch.no_grad():  # 关闭梯度计算
                # 取 encoder 更新出的 embeddings
                embeddings = self.enh_ins_emb
                if len(d.ill_val_idx) > 0:  # 如果有验证集，则使用验证集进行评估
                    result = self.evaluate(it, d.ill_val_idx, embeddings)
                else:  # 否则使用测试集进行评估
                    result = self.evaluate(it, d.ill_test_idx, embeddings)
                
            # Early Stop
            if self.args.early_stop:
                logger.info("Early Stop, patience_cnt = {}, result on test set:".format(patience_cnt))
                if len(self.early_stop_val_result) == 0: # 第一个 epoch
                    self.early_stop_val_result = result
                    self.best_epoch = it
                    with torch.no_grad():
                        self.test_result = self.evaluate(it, d.ill_test_idx, embeddings) if len(d.ill_val_idx) > 0 else result
                else:
                    if result[0][0] >= self.early_stop_val_result[0][0]:
                        patience_cnt = 0
                        self.early_stop_val_result = result
                        self.best_epoch = it
                        with torch.no_grad():
                            self.test_result = self.evaluate(it, d.ill_test_idx, embeddings) if len(d.ill_val_idx) > 0 else result
                    else:
                        patience_cnt += 1
            else:
                self.test_result = result # 不 early_stop 每次记录 result 最后保存的就是最后一个 epoch 的结果

            if self.args.early_stop and patience_cnt > self.args.patience:
                break




    def train_1_epoch(self, it, opt, encoder, decoder, edges, values, rels, triples, ills, ids, ent_emb, rel_emb, others):
        print("train_1_epoch")
        encoder.train()
        decoder.train()
        losses = []
        
        # 判断是否需要更新cached_sample（缓存的样本）
        if decoder.name not in self.cached_sample or it % self.args.update == 0:
            # 根据decoder的类型，设置pos_batch的值
            if decoder.name in ["align", "sinkhorn1", "sinkhorn2", "sinkhorn3", "sinkhorn6"] :
                self.cached_sample[decoder.name] = ills.tolist()
                self.cached_sample[decoder.name] = np.array(self.cached_sample[decoder.name])
            else:
                self.cached_sample[decoder.name] = triples
            np.random.shuffle(self.cached_sample[decoder.name])
        
        # 获取训练样本
        train = self.cached_sample[decoder.name]

        # 设置训练批次大小
        if self.args.train_batch_size == -1:
            train_batch_size = len(train)
        else:
            train_batch_size = self.args.train_batch_size
        # 循环处理每个批次的样本
        for i in range(0, len(train), train_batch_size):
            # 获取正样本
            pos_batch = train[i:i+train_batch_size]

            # 判断是否需要更新cached_sample和进行采样
            if (decoder.name+str(i) not in self.cached_sample or it % self.args.update == 0) and decoder.sampling_method:
                #print("it len(pos_batch, triples, ills), len(ids[0]), self.args.k", it, len(pos_batch), len(triples), len(ills), self.args.k)
                self.cached_sample[decoder.name+str(i)] = decoder.sampling_method(pos_batch, triples, ills, ids, self.args.k, params={
                    "emb": self.enh_ent_embeddings,
                    "metric": self.args.test_dist,
                })


            # 获取负样本
            neg_batch = self.cached_sample[decoder.name+str(i)]

            # 清除梯度
            opt.zero_grad()
            # 构建长度相同的 neg 和 pos
            neg = torch.LongTensor(neg_batch).to(device)

            if neg.size(0) > len(pos_batch) * self.args.k:
                pos = torch.LongTensor(pos_batch).repeat(self.args.k * 2, 1).to(device)
            elif hasattr(decoder.func, "loss"):
                pos = torch.LongTensor(pos_batch).to(device)
            else:
                pos = torch.LongTensor(pos_batch).repeat(self.args.k, 1).to(device)

            # 获取增强嵌入 enh_emb ,即经过encoder的嵌入
            use_edges = torch.LongTensor(edges).to(device)
            use_rels = torch.LongTensor(rels).to(device)

            #print("before encoder")
            if self.args.encoder == "gcn-align":
                enh_emb = encoder.forward(edges=use_edges, rels=None, x=ent_emb, r=rel_emb[d.sparse_rels_idx])
            elif self.args.encoder == "kecg":
                enh_emb = encoder.forward(edges=use_edges, rels=use_rels, x=ent_emb, r=rel_emb)
            elif self.args.encoder == "dual-amn":
                # dual-amn 有额外信息，存储在 others 中
                enh_emb = encoder.forward(edges=use_edges, rels=use_rels, x=ent_emb, r=rel_emb, others=others)
            else:
                enh_emb = encoder.forward(edges=use_edges, rels=use_rels, x=ent_emb, r=rel_emb)
            #print("after encoder")
            # 更新增强实体嵌入 enh_ins_emb
            self.enh_ins_emb =  enh_emb.cpu().detach().numpy()
            #print("self.enh_ins_emb ",self.enh_ins_emb)
            
            # 计算损失
            pos_score = decoder.forward(enh_emb, rel_emb, pos)
            neg_score = decoder.forward(enh_emb, rel_emb, neg)
            target = torch.ones(neg_score.size()).to(device)
            #print("before decoder")
            loss = decoder.loss(pos_score, neg_score, target) * self.args.alpha
            #print("after decoder")
            loss.backward()
            opt.step()
            
            # 记录损失值
            losses.append(loss.item())
        
        # 返回平均损失
        return np.mean(losses)





    def evaluate(self, it, test, ent_emb):
        print("evaluate")
        # 记录评估开始时间
        t_test = time.time()
        # 指定用于计算 top-k 命中率的 k 值

        # 分别取验证/测试集中两个图的实体embedding
        left_emb = ent_emb[test[:, 0]]
        right_emb = ent_emb[test[:, 1]]

        # 计算左实体和右实体之间的欧几里得距离或余弦相似度，并按照距离从小到大排序
        # 先指定为欧几里得距离
        distance = - sim(left_emb, right_emb, metric=self.args.test_dist, normalize=True)

        # 将测试样本分成若干个子集，每个子集由一个进程来处理
        tasks = div_list(np.array(range(len(test))), 10)
        pool = multiprocessing.Pool(processes=len(tasks))
        reses = list()
        for task in tasks:
            # 并行计算 top-k 命中率和平均排名等指标
            reses.append(pool.apply_async(multi_cal_rank, (task, distance[task, :], distance[:, task], self.args.top_k, self.args)))
        pool.close()
        pool.join()
        
        # 合并各个子集的计算结果
        acc_l2r, acc_r2l = np.array([0.] * len(self.args.top_k)), np.array([0.] * len(self.args.top_k))
        mean_l2r, mean_r2l, mrr_l2r, mrr_r2l = 0., 0., 0., 0.
        for res in reses:
            (_acc_l2r, _mean_l2r, _mrr_l2r, _acc_r2l, _mean_r2l, _mrr_r2l) = res.get()
            acc_l2r += _acc_l2r
            mean_l2r += _mean_l2r
            mrr_l2r += _mrr_l2r
            acc_r2l += _acc_r2l
            mean_r2l += _mean_r2l
            mrr_r2l += _mrr_r2l
        mean_l2r /= len(test)
        mean_r2l /= len(test)
        mrr_l2r /= len(test)
        mrr_r2l /= len(test)
        for i in range(len(self.args.top_k)):
            acc_l2r[i] = round(acc_l2r[i] / len(test), 4)
            acc_r2l[i] = round(acc_r2l[i] / len(test), 4)
        
        # 将计算结果记录到 TensorBoard 中
        logger.info("l2r: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.4f} s ".format(self.args.top_k, acc_l2r.tolist(), mean_l2r, mrr_l2r, time.time() - t_test))
        logger.info("r2l: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.4f} s \n".format(self.args.top_k, acc_r2l.tolist(), mean_r2l, mrr_r2l, time.time() - t_test))
        for i, k in enumerate(self.args.top_k):
            writer.add_scalar("l2r_HitsAt{}".format(k), acc_l2r[i], it)
            writer.add_scalar("r2l_HitsAt{}".format(k), acc_r2l[i], it)
        writer.add_scalar("l2r_MeanRank", mean_l2r, it)
        writer.add_scalar("l2r_MeanReciprocalRank", mrr_l2r, it)
        writer.add_scalar("r2l_MeanRank", mean_r2l, it)
        writer.add_scalar("r2l_MeanReciprocalRank", mrr_r2l, it)

        # 返回计算结果
        return (acc_l2r, mean_l2r, mrr_l2r, acc_r2l, mean_r2l, mrr_r2l)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data/DBP15K/zh_en", required=False, help="input dataset file directory, ('data/DBP15K/zh_en', 'data/SRPRS/en_fr_15k_V1')")
    parser.add_argument("--train_rate", type=float, default=0.21, help="training set rate")
    parser.add_argument("--val_rate", type=float, default=0.09, help="valid set rate")

    parser.add_argument("--encoder", type=str, default="GCN-Align", nargs="?", help="which encoder to use")
    parser.add_argument("--hiddens", type=str, default="300,300,300", help="hidden units in each hidden layer(including in_dim and out_dim), splitted with comma")
    parser.add_argument("--heads", type=str, default="1,1", help="heads in each gat layer, splitted with comma")

    parser.add_argument('--skip_conn', type=str, default='none', choices=['none', 'highway', 'concatall', 'concat0andl', 'residual', 'concatallhighway'], help="skip connection")
    parser.add_argument('--ent_init', type=str, default='random', choices=['random', 'name'], help="way to initalize entity embeddings")
    parser.add_argument('--activation', type=str, default='elu', choices=['none', 'elu', 'relu', 'tanh', 'sigmoid'], help="activation function")

    parser.add_argument("--decoder", type=str, default="Align", nargs="?", help="which decoder to use")

    args = parser.parse_args()

    args.hiddens = list(map(int, args.hiddens.split(",")))
    args.heads = list(map(int, args.heads.split(",")))
    args.encoder = args.encoder.lower()
    args.decoder = args.decoder.lower()
    
    
    args.top_k = [1, 3, 5, 10] # metrics: K in Hits@K
    args.cuda = True # whether to use cuda or not
    args.log = "tensorboard_log" # where to save tensorboard the log
    args.seed = 2020 # random seed
    args.epoch = 100 # number of epochs to train
    args.update = 5 # number of epoch for updating negtive samples
    args.train_batch_size = -1 # train batch_size (-1 means all in)
    args.early_stop = True # whether to use early stop 
    args.patience = 5 # early stop patience epoch
    args.bias = False # whether to use bias in encoder.
    args.sampling = "N" # negtive sampling method for each decoder ("N" for nearest neighbor sampling, "T" for typed sampling, "R" for random sampling, "." for no sampling)
    args.k = 25 # negtive sampling number for each decoder
    args.margin = 1.0 # margin for each margin based ranking loss (or params for other loss function)
    args.alpha = 1.0 # weight for each margin based ranking loss
    args.feat_drop = 0.0 # dropout rate for layers
    args.lr = 0.005 # learning rate
    args.train_dist = "euclidean" # distance function used in train (inner, cosine, euclidean, manhattan)
    args.test_dist = "euclidean" # distance function used in test (inner, cosine, euclidean, manhattan)
   
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    writer = SummaryWriter("_runs%s/%s_%s" % (str(time.time()), args.data_dir.split("/")[-1], args.log))
    logger.info(args)

    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    d = AlignmentData(data_dir=args.data_dir, train_rate=args.train_rate, val_rate=args.val_rate)
    logger.info(d)

    experiment = Experiment(args=args)

    t_total = time.time()
    experiment.train_and_eval()

    if args.early_stop:
        logger.info("early stop best epoch: {}, best result:".format(experiment.best_epoch))
    acc_l2r, mean_l2r, mrr_l2r, acc_r2l, mean_r2l, mrr_r2l = experiment.test_result
    logger.info("l2r: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f} ".format(args.top_k, acc_l2r, mean_l2r, mrr_l2r))
    logger.info("r2l: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f} \n".format(args.top_k, acc_r2l, mean_r2l, mrr_r2l))

    logger.info("optimization finished!")
    logger.info("total time elapsed: {:.4f} s".format(time.time() - t_total))
