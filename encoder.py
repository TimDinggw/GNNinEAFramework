import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# --- torch_geometric Packages ---
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, add_remaining_self_loops
from torch_scatter import scatter_add
# --- torch_geometric Packages end ---

from utils import *
from torch_geometric.nn import RGCNConv, SAGEConv, AGNNConv, AntiSymmetricConv, ChebConv, GCNConv, GATv2Conv

# --- Main Models: Encoder ---
class Encoder(torch.nn.Module):
    def __init__(self, name, hiddens, heads, skip_conn, activation, feat_drop, bias, ent_num, rel_num):
        super(Encoder, self).__init__()
        print("encoder init")
        self.name = name
        self.hiddens = hiddens
        self.heads = heads
        self.num_layers = len(hiddens) - 1
        self.gnn_layers = nn.ModuleList()
        self.activation = None
        if activation == 'elu':
            self.activation = F.elu
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "sigmoid":
            self.activation = torch.sigmoid
        self.feat_drop = feat_drop
        self.bias = bias
        self.skip_conn = skip_conn
        self.ent_num = ent_num
        self.rel_num = rel_num
        self.device = None

        self.rgcn_num_bases = 1

        for l in range(0, self.num_layers):
            if self.name == "gcn-align":
                self.gnn_layers.append(
                    GCNAlign_GCNConv(in_channels=self.hiddens[l], out_channels=self.hiddens[l+1], improved=False, cached=True, bias=self.bias)
                )
            elif self.name == "mlp":
                self.gnn_layers.append(
                    nn.Linear(self.hiddens[l], self.hiddens[l+1], bias=self.bias)
                )
            elif self.name == "rgcn":
                self.gnn_layers.append(
                    RGCNConv(self.hiddens[l], self.hiddens[l+1], self.rel_num, bias=self.bias)                    
                )
            elif self.name == "compgcn":
                self.gnn_layers.append(
                    CompGCNConv(self.hiddens[l], self.hiddens[l+1], self.rel_num, self.feat_drop, bias=self.bias)
                )
            elif self.name == "kecg":
                f_in = self.hiddens[l] * self.heads[l - 1] if l else self.hiddens[l]
                self.gnn_layers.append(
                    KECGMultiHeadGraphAttention(self.heads[l], f_in, self.hiddens[l+1], attn_dropout=0.0, init=nn.init.ones_, bias=self.bias)
                )
            elif self.name == "dual-amn":
                self.gnn_layers.append(
                    Dual_AMN_NR_GraphAttention_Layer(node_size=self.ent_num,
                                                    rel_size=self.rel_num,
                                                    node_hidden=self.hiddens[l],
                                                    attn_heads=self.heads[l]),
                )
                self.gnn_layers.append(
                    Dual_AMN_NR_GraphAttention_Layer(node_size=self.ent_num,
                                                    rel_size=self.rel_num,
                                                    node_hidden=self.hiddens[l],
                                                    attn_heads=self.heads[l]),
                )
            elif self.name == "graphsage":
                self.gnn_layers.append(
                    SAGEConv(self.hiddens[l], self.hiddens[l+1], bias=self.bias)
                )
            elif self.name == "agnn":
                self.gnn_layers.append(
                    AGNNConv(add_self_loops = False)
                )
            elif self.name == "antisymmetric":
                self.gnn_layers.append(
                    AntiSymmetricConv(self.hiddens[l])
                )
            elif self.name == "cheb":
                self.gnn_layers.append(
                    ChebConv(self.hiddens[l], self.hiddens[l+1], 1, bias=self.bias)
                )
            elif self.name == "gcn":
                self.gnn_layers.append(
                    GCNConv(self.hiddens[l], self.hiddens[l+1], bias=self.bias)
                )
            elif self.name =="gat":
                self.gnn_layers.append(
                    GATv2Conv(self.hiddens[l], self.hiddens[l+1], bias=self.bias)
                )
            # elif self.name == "SLEF-DESIGN":
            #     self.gnn_layers.append(
            #         SLEF-DESIGN_Conv()
            #     )
            else:
                raise NotImplementedError("bad encoder name: " + self.name)
            
        if self.skip_conn == "highway" or self.skip_conn == "concatallhighway":
            self.gate_weights = nn.ParameterList()
            self.gate_biases = nn.ParameterList()
            for l in range(self.num_layers):
                self.gate_weights.append(nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.hiddens[l], self.hiddens[l]))))    
                self.gate_biases.append(nn.Parameter(torch.zeros(self.hiddens[l])))
        # if self.name == "SLEF-DESIGN":
        #     '''SLEF-DESIGN: extra parameters''
                
    def forward(self, edges, rels, x, r, others=None):
        if self.device is None:
            self.device = x.device
        #print("encoder forward")
        edges = edges.t()

        #print("x before layers", x)

        if self.name == "dual-amn":
            triple_num, ent_adj, rel_adj, r_val, adj_input, r_index = others
            
            # 更新emb
            x = update_node_embeddings(ent_adj, x, self.ent_num, self.hiddens[0])
            print("after x update")

            rel_emb = r
            r = update_node_embeddings(rel_adj, r, self.ent_num, self.hiddens[0])
            print("after r update")

            opt = [rel_emb,adj_input,r_index,r_val]

        all_layer_outputs = [x]

        for l in range(self.num_layers):

            if self.skip_conn == "residual":
                residual = x
                residual = residual.to(self.device)
            if self.skip_conn == "highway" or self.skip_conn == "concatallhighway":
                highway_features = x
                highway_features = highway_features.to(self.device)

            x = F.dropout(x, p=self.feat_drop, training=self.training)

            if self.name == "gcn-align":
                x = self.gnn_layers[l](x, edges)
            elif self.name == "mlp":
                x = self.gnn_layers[l](x)
            elif self.name == "rgcn":
                x = self.gnn_layers[l](x, edges, rels)
            elif self.name == "compgcn":
                x, r = self.gnn_layers[l](x, edges, rels, r)
            elif self.name == "kecg":
                self.diag = True
                x = self.gnn_layers[l](x, edges)
                if self.diag:
                    x = x.mean(dim=0)
            elif self.name == "dual-amn":
                x = self.gnn_layers[l]([x]+opt, triple_num)
                r = self.gnn_layers[l + self.num_layers]([r]+opt, triple_num)
            # elif self.name == "SLEF-DESIGN":
            #     '''SLEF-DESIGN: special encoder forward'''
            else:
                x = self.gnn_layers[l](x, edges) 

            if l != self.num_layers - 1:
                if self.activation:
                    x = self.activation(x)

            if self.skip_conn == "residual":
                x = x + residual

            if self.skip_conn == "highway" or self.skip_conn == "concatallhighway":
                print("highway")
                gate = torch.matmul(highway_features, self.gate_weights[l])
                gate = gate + self.gate_biases[l]
                gate = torch.sigmoid(gate)
                x = x * gate + highway_features * (1.0 - gate)

            if self.name == "dual-amn":
                all_layer_outputs.append(torch.cat([x, r], dim=1))
            else:
                all_layer_outputs.append(x)

        if self.skip_conn == "concatall" or self.skip_conn == "concatallhighway":
            print("concatall")
            return torch.cat(all_layer_outputs, dim=1)
        elif self.skip_conn == 'concat0andl':
            print("concat0andl")
            return torch.cat([all_layer_outputs[0], all_layer_outputs[self.num_layers]], dim=1)
        #print("x after layers", x)
        return x   

    def __repr__(self):
        return '{}(name={}): {}'.format(self.__class__.__name__, self.name, "\n".join([layer.__repr__() for layer in self.gnn_layers]))
# --- Main Models: Encoder end ---

# --- Encoding Modules ---
class GCNAlign_GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, **kwargs):
        super(GCNAlign_GCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.device = None

        self.weight = Parameter(torch.Tensor(1, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        if self.device is None:
            self.device = x.device
        self.weight = self.weight.to(self.device)
        x = torch.mul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class CompGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_rels, feat_drop, bias):
        super(self.__class__, self).__init__()

        self.in_channels	= in_channels
        self.out_channels	= out_channels
        self.num_rels 		= num_rels
        self.device		= None

        self.w_loop		= get_param((in_channels, out_channels))
        self.w_in		= get_param((in_channels, out_channels))
        self.w_out		= get_param((in_channels, out_channels))
        self.w_rel 		= get_param((in_channels, out_channels))
        self.loop_rel 		= get_param((1, in_channels))

        self.drop		= torch.nn.Dropout(feat_drop)
        self.bn			= torch.nn.BatchNorm1d(out_channels)
        self.need_bias = bias
        if self.need_bias: self.register_parameter('bias', Parameter(torch.zeros(out_channels)))

        self.opn = "corr"#Composition Operation to be used in CompGCN

    def forward(self, x, edge_index, edge_type, rel_embed): 
        if self.device is None:
            self.device = edge_index.device

        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
        num_edges = edge_index.size(1) // 2
        num_ent   = x.size(0)

        self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
        self.in_type,  self.out_type  = edge_type[:num_edges], 	 edge_type [num_edges:]

        self.loop_index  = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
        self.loop_type   = torch.full((num_ent,), rel_embed.size(0)-1, dtype=torch.long).to(self.device)

        self.in_norm     = self.compute_norm(self.in_index,  num_ent)
        self.out_norm    = self.compute_norm(self.out_index, num_ent)

        in_res		= self.propagate(aggr='add', edge_index=self.in_index,   x=x, edge_type=self.in_type,   rel_embed=rel_embed, edge_norm=self.in_norm, 	mode='in')
        loop_res	= self.propagate(aggr='add', edge_index=self.loop_index, x=x, edge_type=self.loop_type, rel_embed=rel_embed, edge_norm=None, 		mode='loop')
        out_res		= self.propagate(aggr='add', edge_index=self.out_index,  x=x, edge_type=self.out_type,  rel_embed=rel_embed, edge_norm=self.out_norm,	mode='out')
        out		= self.drop(in_res)*(1/3) + self.drop(out_res)*(1/3) + loop_res*(1/3)

        if self.need_bias: out = out + self.bias
        out = self.bn(out)

        return out, torch.matmul(rel_embed, self.w_rel)[:-1]		# Ignoring the self loop inserted

    def rel_transform(self, ent_embed, rel_embed):
        if   self.opn == 'corr': 	trans_embed  = ccorr(ent_embed, rel_embed)
        elif self.opn == 'sub': 	trans_embed  = ent_embed - rel_embed
        elif self.opn == 'mult': 	trans_embed  = ent_embed * rel_embed
        else: raise NotImplementedError

        return trans_embed

    def message(self, x_j, edge_type, rel_embed, edge_norm, mode):
        weight 	= getattr(self, 'w_{}'.format(mode))
        rel_emb = torch.index_select(rel_embed, 0, edge_type)
        xj_rel  = self.rel_transform(x_j, rel_emb)
        out	= torch.mm(xj_rel, weight)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out

    def compute_norm(self, edge_index, num_ent):
        row, col	= edge_index
        edge_weight 	= torch.ones_like(row).float()
        deg		= scatter_add( edge_weight, row, dim=0, dim_size=num_ent)	# Summing number of weights of the edges
        deg_inv		= deg.pow(-0.5)							# D^{-0.5}
        deg_inv[deg_inv	== float('inf')] = 0
        norm		= deg_inv[row] * edge_weight * deg_inv[col]			# D^{-0.5}

        return norm

    def __repr__(self):
        return '{}({}, {}, num_rels={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)    


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b

class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

class KECGMultiHeadGraphAttention(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    https://github.com/Diego999/pyGAT/blob/master/layers.py
    """
    def __init__(self, n_head, f_in, f_out, attn_dropout, diag=True, init=None, bias=False):
        super(KECGMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.diag = diag
        self.device = None
        if self.diag:
            self.w = Parameter(torch.Tensor(n_head, 1, f_out))
        else:
            self.w = Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src_dst = Parameter(torch.Tensor(n_head, f_out * 2, 1))
        self.attn_dropout = attn_dropout
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.special_spmm = SpecialSpmm()
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)
        if init is not None and diag:
            init(self.w)
            stdv = 1. / math.sqrt(self.a_src_dst.size(1))
            nn.init.uniform_(self.a_src_dst, -stdv, stdv)
        else:
            nn.init.xavier_uniform_(self.w)
            nn.init.xavier_uniform_(self.a_src_dst)

    def forward(self, input, edge):
        if self.device is None:
            self.device = input.device
        self.w = self.w.to(self.device)
        if self.bias is not None:
            self.bias = self.bias.to(self.device)
        output = []
        for i in range(self.n_head):
            N = input.size()[0]
            if self.diag:
                h = torch.mul(input, self.w[i])
            else:
                h = torch.mm(input, self.w[i])

            edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1) # edge: 2*D x E
            edge_e = torch.exp(-self.leaky_relu(edge_h.mm(self.a_src_dst[i]).squeeze())) # edge_e: 1 x E
            e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1)).cuda()  if next(self.parameters()).is_cuda else torch.ones(size=(N, 1))) # e_rowsum: N x 1
            edge_e = F.dropout(edge_e, self.attn_dropout, training=self.training)   # edge_e: 1 x E
            h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
            h_prime = h_prime.div(e_rowsum)
            output.append(h_prime.unsqueeze(0))
        
        output = torch.cat(output, dim=0)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    
    def __repr__(self):
        if self.diag:
            return self.__class__.__name__ + ' (' + str(self.f_out) + ' -> ' + str(self.f_out) + ') * ' + str(self.n_head) + ' heads'
        else:
            return self.__class__.__name__ + ' (' + str(self.f_in) + ' -> ' + str(self.f_out) + ') * ' + str(self.n_head) + ' heads'


class Dual_AMN_NR_GraphAttention_Layer(nn.Module):
    def __init__(self,
            node_size,
            rel_size,
            node_hidden,
            attn_heads=1):
        super(Dual_AMN_NR_GraphAttention_Layer, self).__init__()
        self.node_size = node_size
        self.rel_size = rel_size
        self.node_hidden = node_hidden
        self.attn_heads = attn_heads
        self.attn_kernels = []
        self.device = None

        for head in range(self.attn_heads):
            attn_kernel = Parameter(torch.Tensor(1 * self.node_hidden, 1))
            self.attn_kernels.append(attn_kernel)

        for head in range(self.attn_heads):
            nn.init.xavier_uniform_(self.attn_kernels[head])

    def forward(self, inputs, triple_size):
        features = inputs[0]
        rel_emb = inputs[1]
        if self.device is None:
            self.device = features.device
        indices = inputs[2].t().to(torch.long).to(self.device)
        sparse_indices = inputs[3]
        sparse_val = inputs[4]

        features_list = []
        for head in range(self.attn_heads):
            attention_kernel = self.attn_kernels[head].to(self.device)
            rels_sum = torch.sparse_coo_tensor(sparse_indices.t(), sparse_val,
                                            (triple_size, self.rel_size))

            rels_sum = rels_sum.to(torch.float32).to(self.device)
            rel_emb_weight = rel_emb.to(torch.float32)
            rels_sum = torch.sparse.mm(rels_sum, rel_emb_weight)
            print("indices", indices)
            print("features", features)
            neighs = features[indices[1]]
            selfs = features[indices[0]]

            rels_sum = F.normalize(rels_sum, dim=1)
            print("indices",indices.shape)
            print("neighs,rels_sum",neighs.shape, rels_sum.shape)
            print("attention_kernel", attention_kernel)
            neighs = neighs - 2 * torch.sum(neighs * rels_sum, 1, keepdim=True) * rels_sum        
            att = torch.squeeze(torch.matmul(rels_sum, attention_kernel), dim=-1)

            # 提取每个非零元素所在的行
            rows = indices[0]
            # 对每一行的非零元素值应用softmax
            exp_values = att.exp()
            row_sums = torch.zeros(self.node_size).to(features.device)
            row_sums.index_add_(0, rows, exp_values)  # 计算每行的指数和
            softmax_values = exp_values / (row_sums[rows])  # 应用softmax
            neighs_att = neighs * softmax_values.unsqueeze(-1)

            # 获取 adj_indices 的第一列作为 segment_ids
            segment_ids = indices[0]

            # 初始化新的特征张量 new_features
            new_features = torch.zeros(self.node_size, neighs_att.size(-1))

            # 对 neighs_att 根据 segment_ids 进行分段求和
            for i in range(self.node_size):
                mask = (segment_ids == i)
                new_features[i] = torch.sum(neighs_att[mask], dim=0)
            features_list.append(new_features)

        features = torch.mean(torch.stack(features_list), dim=0)
        print("final features", features)
        return features

# class SELF-DESIGN-ENCODER(torch.nn.Module):
#    '''SELF-DESIGN-ENCODER:implement __init__, forward, ...'''

# --- Encoding Modules end ---
