import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool as gmp, global_mean_pool
from torch_geometric.nn import GATv2Conv  # Using GATv2Conv instead of GATConv
from torch_scatter import scatter_mean
from torch.nn import Parameter
import numpy as np

EPS = 1e-15

def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()
    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


class NodeRepresentation(nn.Module):
    def __init__(self, gcn_layer, dim_gexp, dim_methy, output, units_list=[256, 256, 256], 
                 use_relu=True, use_bn=True, use_GMP=True, use_mutation=True, 
                 use_gexpr=True, use_methylation=True, dropout=0.2, num_heads=4):
        super(NodeRepresentation, self).__init__()
        torch.manual_seed(0)
        self.use_relu = use_relu
        self.use_bn = use_bn
        self.units_list = units_list
        self.use_GMP = use_GMP
        self.use_mutation = use_mutation
        self.use_gexpr = use_gexpr
        self.use_methylation = use_methylation
        self.dropout = dropout
        self.num_heads = num_heads
        
        # Using GATv2Conv instead of GATConv for dynamic attention
        self.conv1 = GATv2Conv(gcn_layer, units_list[0] // num_heads, heads=num_heads, dropout=dropout)
        self.batch_conv1 = nn.BatchNorm1d(units_list[0])
        self.graph_conv = nn.ModuleList()
        self.graph_bn = nn.ModuleList()
        
        for i in range(len(units_list) - 1):
            self.graph_conv.append(
                GATv2Conv(units_list[i], units_list[i + 1] // num_heads, heads=num_heads, dropout=dropout)
            )
            self.graph_bn.append(nn.BatchNorm1d(units_list[i + 1]))
        
        self.conv_end = GATv2Conv(units_list[-1], output // num_heads, heads=num_heads, dropout=dropout)
        self.batch_end = nn.BatchNorm1d(output)
        
        # Cell line layers with dropout
        self.fc_gexp1 = nn.Linear(dim_gexp, 512)
        self.batch_gexp1 = nn.BatchNorm1d(512)
        self.fc_gexp2 = nn.Linear(512, 256)
        self.batch_gexp2 = nn.BatchNorm1d(256)
        self.fc_gexp3 = nn.Linear(256, output)
        
        self.fc_methy1 = nn.Linear(dim_methy, 512)
        self.batch_methy1 = nn.BatchNorm1d(512)
        self.fc_methy2 = nn.Linear(512, 256)
        self.batch_methy2 = nn.BatchNorm1d(256)
        self.fc_methy3 = nn.Linear(256, output)
        
        # Improved CNN architecture for mutation data
        self.cov1 = nn.Conv2d(1, 64, (1, 7), stride=(1, 3))
        self.bn_cov1 = nn.BatchNorm2d(64)
        self.cov2 = nn.Conv2d(64, 32, (1, 5), stride=(1, 2))
        self.bn_cov2 = nn.BatchNorm2d(32)
        self.fla_mut = nn.Flatten()
        self.fc_mut1 = nn.Linear(2144, 512)  # Adjusted size based on new conv layers
        self.bn_mut1 = nn.BatchNorm1d(512)
        self.fc_mut2 = nn.Linear(512, output)
        
        # Feature combination layers
        self.fcat = nn.Linear(300, 200)
        self.bn_fcat = nn.BatchNorm1d(200)
        self.fcat2 = nn.Linear(200, output)
        self.batchc = nn.BatchNorm1d(output)
        
        # Dropout layers
        self.dropout_layer = nn.Dropout(dropout)
        
        self.reset_para()

    def reset_para(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, drug_feature, drug_adj, ibatch, mutation_data, gexpr_data, methylation_data):
        # Drug graph processing with improved GAT
        x_drug = self.conv1(drug_feature, drug_adj)
        x_drug = F.leaky_relu(x_drug, negative_slope=0.2)
        x_drug = self.dropout_layer(x_drug)
        x_drug = self.batch_conv1(x_drug)
        
        for i in range(len(self.units_list) - 1):
            x_drug = self.graph_conv[i](x_drug, drug_adj)
            x_drug = F.leaky_relu(x_drug, negative_slope=0.2)
            x_drug = self.dropout_layer(x_drug)
            x_drug = self.graph_bn[i](x_drug)
        
        x_drug = self.conv_end(x_drug, drug_adj)
        x_drug = F.leaky_relu(x_drug, negative_slope=0.2)
        x_drug = self.batch_end(x_drug)
        
        # Global pooling with residual connection
        if self.use_GMP:
            x_drug_mean = global_mean_pool(x_drug, ibatch)
            x_drug_max = gmp(x_drug, ibatch)
            x_drug = x_drug_mean + x_drug_max
        else:
            x_drug = global_mean_pool(x_drug, ibatch)
        
        # Omics data processing with deeper networks
        outputs = []
        
        if self.use_mutation:
            x_mutation = self.cov1(mutation_data)
            x_mutation = self.bn_cov1(x_mutation)
            x_mutation = F.leaky_relu(x_mutation, negative_slope=0.2)
            x_mutation = F.max_pool2d(x_mutation, (1, 3))
            x_mutation = self.dropout_layer(x_mutation)
            
            x_mutation = self.cov2(x_mutation)
            x_mutation = self.bn_cov2(x_mutation)
            x_mutation = F.leaky_relu(x_mutation, negative_slope=0.2)
            x_mutation = F.max_pool2d(x_mutation, (1, 3))
            x_mutation = self.dropout_layer(x_mutation)
            
            x_mutation = self.fla_mut(x_mutation)
            x_mutation = self.fc_mut1(x_mutation)
            x_mutation = self.bn_mut1(x_mutation)
            x_mutation = F.leaky_relu(x_mutation, negative_slope=0.2)
            x_mutation = self.dropout_layer(x_mutation)
            x_mutation = self.fc_mut2(x_mutation)
            
            outputs.append(x_mutation)

        if self.use_gexpr:
            x_gexpr = self.fc_gexp1(gexpr_data)
            x_gexpr = self.batch_gexp1(x_gexpr)
            x_gexpr = F.leaky_relu(x_gexpr, negative_slope=0.2)
            x_gexpr = self.dropout_layer(x_gexpr)
            
            x_gexpr = self.fc_gexp2(x_gexpr)
            x_gexpr = self.batch_gexp2(x_gexpr)
            x_gexpr = F.leaky_relu(x_gexpr, negative_slope=0.2)
            x_gexpr = self.dropout_layer(x_gexpr)
            
            x_gexpr = self.fc_gexp3(x_gexpr)
            
            outputs.append(x_gexpr)

        if self.use_methylation:
            x_methylation = self.fc_methy1(methylation_data)
            x_methylation = self.batch_methy1(x_methylation)
            x_methylation = F.leaky_relu(x_methylation, negative_slope=0.2)
            x_methylation = self.dropout_layer(x_methylation)
            
            x_methylation = self.fc_methy2(x_methylation)
            x_methylation = self.batch_methy2(x_methylation)
            x_methylation = F.leaky_relu(x_methylation, negative_slope=0.2)
            x_methylation = self.dropout_layer(x_methylation)
            
            x_methylation = self.fc_methy3(x_methylation)
            
            outputs.append(x_methylation)

        if outputs:
            x_cell = torch.cat(outputs, 1)
            x_cell = self.fcat(x_cell)
            x_cell = self.bn_fcat(x_cell)
            x_cell = F.leaky_relu(x_cell, negative_slope=0.2)
            x_cell = self.dropout_layer(x_cell)
            x_cell = self.fcat2(x_cell)
        else:
            x_cell = torch.zeros(drug_feature.size(0), self.hidden_size, device=drug_feature.device)
        
        x_all = torch.cat((x_cell, x_drug), 0)
        x_all = self.batchc(x_all)
        return x_all


class GATv2Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads=4, dropout=0.2):
        super(GATv2Encoder, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout)
        self.prelu1 = nn.PReLU(hidden_channels)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout)
        self.prelu2 = nn.PReLU(hidden_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.prelu1(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.prelu2(x)
        return x


class AttentionSummary(nn.Module):
    def __init__(self, ino, inn):
        super(AttentionSummary, self).__init__()
        self.fc1 = nn.Linear(ino + inn, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, xo, xn):
        # Multi-layer attention mechanism
        m = torch.cat((xo, xn), 1)
        m = self.fc1(m)
        m = self.bn1(m)
        m = F.leaky_relu(m, negative_slope=0.2)
        m = self.dropout(m)
        
        m = self.fc2(m)
        m = self.bn2(m)
        m = F.leaky_relu(m, negative_slope=0.2)
        m = self.dropout(m)
        
        m = self.fc3(m)
        m = torch.tanh(torch.squeeze(m))
        
        # Softmax attention weights
        attention = F.softmax(m, dim=0)
        
        # Weighted sum
        x = torch.matmul(attention, xn)
        return x


class ImprovedGraphCDR(nn.Module):
    def __init__(self, hidden_channels, encoder, summary, feat, index, dropout=0.2):
        super(ImprovedGraphCDR, self).__init__()
        self.hidden_channels = hidden_channels
        self.encoder = encoder
        self.summary = summary
        self.feat = feat
        self.index = index
        self.weight = Parameter(torch.Tensor(hidden_channels, hidden_channels))
        self.act = nn.Sigmoid()
        
        # Deeper prediction layers
        self.fc = nn.Linear(100, 64)
        self.bn_fc = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 10)
        
        self.fd = nn.Linear(100, 64)
        self.bn_fd = nn.BatchNorm1d(64)
        self.fd2 = nn.Linear(64, 10)
        
        # Layer normalization for better training stability
        self.layer_norm = nn.LayerNorm(hidden_channels)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.summary)
        nn.init.xavier_uniform_(self.weight)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, drug_feature, drug_adj, ibatch, mutation_data, gexpr_data, methylation_data, edge):
        # CDR graph edge and corrupted CDR graph edge
        pos_edge = torch.from_numpy(edge[edge[:, 2] == 1, 0:2].T)
        neg_edge = torch.from_numpy(edge[edge[:, 2] == -1, 0:2].T)
        
        # Cell+drug node attributes
        feature = self.feat(drug_feature, drug_adj, ibatch, mutation_data, gexpr_data, methylation_data)
        
        # Cell+drug embedding from the CDR graph and the corrupted CDR graph
        pos_z = self.encoder(feature, pos_edge)
        neg_z = self.encoder(feature, neg_edge)
        
        # Layer normalization for stable training
        pos_z = self.layer_norm(pos_z)
        neg_z = self.layer_norm(neg_z)
        
        # Apply dropout for regularization
        pos_z = self.dropout(pos_z)
        neg_z = self.dropout(neg_z)
        
        # Graph-level embedding (summary)
        summary_pos = self.summary(feature, pos_z)
        summary_neg = self.summary(feature, neg_z)
        
        # Embedding at layer l
        cellpos = pos_z[:self.index, ]
        drugpos = pos_z[self.index:, ]
        
        # Embedding at layer 0 with deeper networks
        cell_features = feature[:self.index, ]
        drug_features = feature[self.index:, ]
        
        cellfea = self.fc(cell_features)
        cellfea = self.bn_fc(cellfea)
        cellfea = F.leaky_relu(cellfea, negative_slope=0.2)
        cellfea = self.dropout(cellfea)
        cellfea = self.fc2(cellfea)
        cellfea = torch.sigmoid(cellfea)
        
        drugfea = self.fd(drug_features)
        drugfea = self.bn_fd(drugfea)
        drugfea = F.leaky_relu(drugfea, negative_slope=0.2)
        drugfea = self.dropout(drugfea)
        drugfea = self.fd2(drugfea)
        drugfea = torch.sigmoid(drugfea)
        
        # Concatenate embeddings at different layers (0 and l)
        cellpos = torch.cat((cellpos, cellfea), 1)
        drugpos = torch.cat((drugpos, drugfea), 1)
        
        # Inner product with scaled attention
        pos_adj = torch.matmul(cellpos, drugpos.t()) / np.sqrt(cellpos.size(1))
        pos_adj = self.act(pos_adj)
        
        return pos_z, neg_z, summary_pos, summary_neg, pos_adj.view(-1)

    def discriminate(self, z, summary, sigmoid=True):
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value

    def loss(self, pos_z, neg_z, summary):
        # Weighted loss with regularization
        pos_loss = -torch.log(self.discriminate(pos_z, summary, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(1 - self.discriminate(neg_z, summary, sigmoid=True) + EPS).mean()
        
        # L2 regularization
        l2_reg = 0.001 * (torch.norm(self.weight) + torch.sum(torch.stack([torch.norm(p) for p in self.parameters()])))
        
        return pos_loss + neg_loss + l2_reg

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.hidden_channels)


# Usage Example
def create_model(gcn_input_dim, dim_gexp, dim_methy, cell_count, dropout=0.2):
    # Create the NodeRepresentation module
    node_rep = NodeRepresentation(
        gcn_layer=gcn_input_dim,
        dim_gexp=dim_gexp,
        dim_methy=dim_methy,
        output=100,
        units_list=[256, 256, 256],
        dropout=dropout,
        num_heads=4
    )
    
    # Create the Encoder
    encoder = GATv2Encoder(100, 100, num_heads=4, dropout=dropout)
    
    # Create the Summary module
    summary = AttentionSummary(100, 100)
    
    # Create the GraphCDR model
    model = ImprovedGraphCDR(100, encoder, summary, node_rep, cell_count, dropout=dropout)
    
    return model
