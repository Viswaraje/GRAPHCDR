import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool as gmp, global_mean_pool, GATv2Conv
from torch_geometric.nn import BatchNorm
from torch.nn import Parameter
import numpy as np

EPS = 1e-15

def reset(nn):
    """Reset parameters using Xavier initialization."""
    for m in nn.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def glorot(tensor):
    """Initialize tensor with Glorot initialization."""
    if tensor is not None:
        stdv = np.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

class NodeRepresentation(nn.Module):
    def __init__(self, gcn_layer, dim_gexp, dim_methy, output, units_list=[256, 256, 256], 
                 dropout=0.2, use_relu=True, use_bn=True, use_GMP=True, 
                 use_mutation=True, use_gexpr=True, use_methylation=True):
        super(NodeRepresentation, self).__init__()
        torch.manual_seed(42)  # Consistent seed for reproducibility
        
        # Model parameters
        self.use_relu = use_relu
        self.use_bn = use_bn
        self.units_list = units_list
        self.use_GMP = use_GMP
        self.use_mutation = use_mutation
        self.use_gexpr = use_gexpr
        self.use_methylation = use_methylation
        self.dropout = dropout
        
        # Use GATv2Conv instead of GATConv for better performance
        self.conv1 = GATv2Conv(gcn_layer, units_list[0], heads=4, dropout=dropout, concat=False)
        self.batch_conv1 = nn.BatchNorm1d(units_list[0])
        
        # Graph neural network layers
        self.graph_conv = nn.ModuleList()
        self.graph_bn = nn.ModuleList()
        for i in range(len(units_list) - 1):
            self.graph_conv.append(GATv2Conv(units_list[i], units_list[i + 1], heads=4, dropout=dropout, concat=False))
            self.graph_bn.append(nn.BatchNorm1d((units_list[i + 1])))
        
        self.conv_end = GATv2Conv(units_list[-1], output, heads=1, concat=False)
        self.batch_end = nn.BatchNorm1d(output)
        
        # Gene expression layers with residual connections
        self.fc_gexp1 = nn.Linear(dim_gexp, 512)
        self.batch_gexp1 = nn.BatchNorm1d(512)
        self.fc_gexp2 = nn.Linear(512, 256)
        self.batch_gexp2 = nn.BatchNorm1d(256)
        self.fc_gexp3 = nn.Linear(256, output)
        
        # Methylation layers with residual connections
        self.fc_methy1 = nn.Linear(dim_methy, 512)
        self.batch_methy1 = nn.BatchNorm1d(512)
        self.fc_methy2 = nn.Linear(512, 256)
        self.batch_methy2 = nn.BatchNorm1d(256)
        self.fc_methy3 = nn.Linear(256, output)
        
        # Mutation layers - improved CNN architecture
        self.cov1 = nn.Conv2d(1, 64, (1, 9), stride=(1, 3))
        self.bn_cov1 = nn.BatchNorm2d(64)
        self.cov2 = nn.Conv2d(64, 32, (1, 5), stride=(1, 2))
        self.bn_cov2 = nn.BatchNorm2d(32)
        self.fla_mut = nn.Flatten()
        self.fc_mut1 = nn.Linear(2048, 512)  # Adjust this size based on your input dimensions
        self.bn_mut1 = nn.BatchNorm1d(512)
        self.fc_mut2 = nn.Linear(512, output)
        
        # Concatenate three omics with attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(output * 3, 3),
            nn.Softmax(dim=1)
        )
        
        # Final combination layers
        self.fcat = nn.Linear(output, output)
        self.batchc = nn.BatchNorm1d(output)
        
        # Cell-drug interaction layer
        self.interaction = nn.Sequential(
            nn.Linear(output * 2, output),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output, output // 2),
            nn.ReLU(),
            nn.BatchNorm1d(output // 2)
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize model parameters"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, drug_feature, drug_adj, ibatch, mutation_data, gexpr_data, methylation_data):
        # Prepare edge_index for GAT
        num_nodes = drug_feature.size(0)
        
        # Convert adjacency matrix to edge_index for GAT
        if isinstance(drug_adj, torch.Tensor) and len(drug_adj.shape) == 2:
            # Make sure adj matrix dimensions match feature dimensions
            if drug_adj.shape[0] > num_nodes or drug_adj.shape[1] > num_nodes:
                drug_adj = drug_adj[:num_nodes, :num_nodes]
            
            edge_index = torch.nonzero(drug_adj).t().contiguous()
            
            # Safety check - filter out any edge indices that are out of bounds
            valid_edges_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
            edge_index = edge_index[:, valid_edges_mask]
        else:
            # If drug_adj is already an edge_index (PyG format)
            edge_index = drug_adj
            # Safety check - filter out any edge indices that are out of bounds
            valid_edges_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
            edge_index = edge_index[:, valid_edges_mask]
        
        # Ensure edge_index is not empty
        if edge_index.numel() == 0:
            # Create a self-loop for each node as a fallback
            edge_index = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)], dim=0).to(drug_feature.device)
       
        # -----drug representation
        x_drug = self.conv1(drug_feature, edge_index)
        x_drug = F.leaky_relu(x_drug, 0.2)
        x_drug = F.dropout(x_drug, p=self.dropout, training=self.training)
        x_drug = self.batch_conv1(x_drug)
        
        for i in range(len(self.units_list) - 1):
            x_res = x_drug  # Store for residual connection
            x_drug = self.graph_conv[i](x_drug, edge_index)
            x_drug = F.leaky_relu(x_drug, 0.2)
            x_drug = F.dropout(x_drug, p=self.dropout, training=self.training)
            x_drug = self.graph_bn[i](x_drug)
            # Add residual connection if shapes match
            if x_drug.shape == x_res.shape:
                x_drug = x_drug + x_res
        
        x_drug = self.conv_end(x_drug, edge_index)
        x_drug = F.leaky_relu(x_drug, 0.2)
        x_drug = self.batch_end(x_drug)
        
        if self.use_GMP:
            x_drug = gmp(x_drug, ibatch)
        else:
            x_drug = global_mean_pool(x_drug, ibatch)
        
        # -----cell line representation
        # -----mutation representation
        if self.use_mutation:
            x_mutation = self.cov1(mutation_data)
            x_mutation = self.bn_cov1(x_mutation)
            x_mutation = F.leaky_relu(x_mutation, 0.2)
            x_mutation = F.max_pool2d(x_mutation, (1, 3))
            x_mutation = F.dropout(x_mutation, p=self.dropout, training=self.training)
            
            x_mutation = self.cov2(x_mutation)
            x_mutation = self.bn_cov2(x_mutation)
            x_mutation = F.leaky_relu(x_mutation, 0.2)
            x_mutation = F.max_pool2d(x_mutation, (1, 3))
            x_mutation = F.dropout(x_mutation, p=self.dropout, training=self.training)
            
            x_mutation = self.fla_mut(x_mutation)
            x_mutation = self.fc_mut1(x_mutation)
            x_mutation = self.bn_mut1(x_mutation)
            x_mutation = F.leaky_relu(x_mutation, 0.2)
            x_mutation = F.dropout(x_mutation, p=self.dropout, training=self.training)
            x_mutation = self.fc_mut2(x_mutation)
        
        # ----gene expression representation
        if self.use_gexpr:
            x_gexpr = self.fc_gexp1(gexpr_data)
            x_gexpr = self.batch_gexp1(x_gexpr)
            x_gexpr = F.leaky_relu(x_gexpr, 0.2)
            x_gexpr = F.dropout(x_gexpr, p=self.dropout, training=self.training)
            
            x_gexpr = self.fc_gexp2(x_gexpr)
            x_gexpr = self.batch_gexp2(x_gexpr)
            x_gexpr = F.leaky_relu(x_gexpr, 0.2)
            x_gexpr = F.dropout(x_gexpr, p=self.dropout, training=self.training)
            
            x_gexpr = self.fc_gexp3(x_gexpr)
        
        # ----methylation representation
        if self.use_methylation:
            x_methylation = self.fc_methy1(methylation_data)
            x_methylation = self.batch_methy1(x_methylation)
            x_methylation = F.leaky_relu(x_methylation, 0.2)
            x_methylation = F.dropout(x_methylation, p=self.dropout, training=self.training)
            
            x_methylation = self.fc_methy2(x_methylation)
            x_methylation = self.batch_methy2(x_methylation)
            x_methylation = F.leaky_relu(x_methylation, 0.2)
            x_methylation = F.dropout(x_methylation, p=self.dropout, training=self.training)
            
            x_methylation = self.fc_methy3(x_methylation)
        
        # Calculate how many omics are used for concatenation
        omics_used = 0
        omics_tensors = []
        
        if self.use_mutation:
            omics_used += 1
            omics_tensors.append(x_mutation)
        if self.use_gexpr:
            omics_used += 1
            omics_tensors.append(x_gexpr)
        if self.use_methylation:
            omics_used += 1
            omics_tensors.append(x_methylation)
        
        # Apply attention mechanism to fuse omics data if we have multiple sources
        if omics_used > 1:
            omics_stack = torch.stack(omics_tensors, dim=1)  # [batch_size, omics_count, output_dim]
            # Apply attention weights
            attention_weights = self.attention(torch.cat(omics_tensors, dim=1))
            attention_weights = attention_weights.unsqueeze(2)  # [batch_size, omics_count, 1]
            x_cell = torch.sum(omics_stack * attention_weights, dim=1)  # Weighted sum
        elif omics_used == 1:
            x_cell = omics_tensors[0]
        else:
            # Fallback if no omics data is used (unlikely case)
            x_cell = torch.zeros_like(x_drug)
        
        # Final cell representation
        x_cell = F.leaky_relu(self.fcat(x_cell), 0.2)
        
        # Combine all representations
        x_all = torch.cat((x_cell, x_drug), 0)
        x_all = self.batchc(x_all)
        
        return x_all

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.2):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=4, dropout=dropout, concat=False))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Additional layers
        for i in range(num_layers - 1):
            self.convs.append(GATv2Conv(hidden_channels, hidden_channels, heads=4, dropout=dropout, concat=False))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Skip connection for residual learning
        self.skip = nn.Linear(in_channels, hidden_channels) if in_channels != hidden_channels else nn.Identity()
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for conv in self.convs:
            if hasattr(conv, 'reset_parameters'):
                conv.reset_parameters()
        for bn in self.bns:
            if hasattr(bn, 'reset_parameters'):
                bn.reset_parameters()
        if isinstance(self.skip, nn.Linear):
            nn.init.xavier_uniform_(self.skip.weight)
            if self.skip.bias is not None:
                nn.init.zeros_(self.skip.bias)
    
    def forward(self, x, edge_index):
        # Safety check for edge_index
        num_nodes = x.size(0)
        valid_edges_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
        edge_index = edge_index[:, valid_edges_mask]
        
        # Ensure edge_index is not empty
        if edge_index.numel() == 0:
            # Create a self-loop for each node as a fallback
            edge_index = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)], dim=0).to(x.device)
        
        # Skip connection
        h = self.skip(x)
        
        # Apply layers with residual connections
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.leaky_relu(x, 0.2)
            x = self.bns[i](x)
            
            # Add residual connection after first layer
            if i == 0:
                x = x + h
        
        return x

class Summary(nn.Module):
    def __init__(self, ino, inn, hidden_dim=64):
        super(Summary, self).__init__()
        # Multi-layer attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(ino + inn, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )
        self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, xo, xn):
        # Improved attention mechanism
        batch_size = xo.size(0)
        n_nodes = xn.size(0)
        
        # Expand xo to match with each node in xn
        xo_expanded = xo.unsqueeze(1).expand(batch_size, n_nodes, -1)
        xn_expanded = xn.unsqueeze(0).expand(batch_size, n_nodes, -1)
        
        # Concatenate node features with global features
        cat_features = torch.cat((xo_expanded, xn_expanded), dim=2).view(batch_size * n_nodes, -1)
        
        # Calculate attention scores
        att_scores = self.attention(cat_features).view(batch_size, n_nodes)
        att_weights = F.softmax(att_scores, dim=1)
        
        # Apply attention weights to node features
        weighted_nodes = xn_expanded * att_weights.unsqueeze(2)
        
        # Sum to get graph-level embedding
        graph_embedding = weighted_nodes.sum(dim=1)
        
        return graph_embedding

class GraphCDR(nn.Module):
    def __init__(self, hidden_channels, encoder, summary, feat, index, dropout=0.2):
        super(GraphCDR, self).__init__()
        self.hidden_channels = hidden_channels
        self.encoder = encoder
        self.summary = summary
        self.feat = feat
        self.index = index
        self.dropout = dropout
        
        # Multi-layer prediction head
        self.weight = Parameter(torch.Tensor(hidden_channels, hidden_channels))
        self.weight2 = Parameter(torch.Tensor(hidden_channels, hidden_channels))
        
        # Improved prediction layers
        self.fc = nn.Sequential(
            nn.Linear(100, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2)
        )
        
        self.fd = nn.Sequential(
            nn.Linear(100, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2)
        )
        
        # Final prediction layers
        self.predictor = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1)
        )
        
        self.act = nn.Sigmoid()
        self.reset_parameters()
    
    def reset_parameters(self):
        reset(self.encoder)
        reset(self.summary)
        glorot(self.weight)
        glorot(self.weight2)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, drug_feature, drug_adj, ibatch, mutation_data, gexpr_data, methylation_data, edge):
        # Extract positive and negative edges
        pos_edge = torch.from_numpy(edge[edge[:, 2] == 1, 0:2].T)
        neg_edge = torch.from_numpy(edge[edge[:, 2] == -1, 0:2].T)
        
        # Get node features
        feature = self.feat(drug_feature, drug_adj, ibatch, mutation_data, gexpr_data, methylation_data)
        
        # Safety check for edge indices
        num_nodes = feature.size(0)
        
        # Filter valid edges for pos_edge
        valid_pos_mask = (pos_edge[0] < num_nodes) & (pos_edge[1] < num_nodes)
        pos_edge = pos_edge[:, valid_pos_mask]
        
        # Filter valid edges for neg_edge
        valid_neg_mask = (neg_edge[0] < num_nodes) & (neg_edge[1] < num_nodes)
        neg_edge = neg_edge[:, valid_neg_mask]
        
        # Ensure edges are not empty
        if pos_edge.numel() == 0:
            pos_edge = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)], dim=0).to(feature.device)
        if neg_edge.numel() == 0:
            neg_edge = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)], dim=0).to(feature.device)
        
        # Get embeddings
        pos_z = self.encoder(feature, pos_edge)
        neg_z = self.encoder(feature, neg_edge)
        
        # Apply dropout for regularization
        pos_z = F.dropout(pos_z, p=self.dropout, training=self.training)
        neg_z = F.dropout(neg_z, p=self.dropout, training=self.training)
        
        # Get graph-level embeddings
        summary_pos = self.summary(feature, pos_z)
        summary_neg = self.summary(feature, neg_z)
        
        # Get embeddings for cells and drugs
        cellpos = pos_z[:self.index, ]
        drugpos = pos_z[self.index:, ]
        
        # Process initial features
        cellfea = self.fc(feature[:self.index, ])
        drugfea = self.fd(feature[self.index:, ])
        
        # Concatenate embeddings from different layers
        cell_combined = torch.cat((cellpos, cellfea), 1)
        drug_combined = torch.cat((drugpos, drugfea), 1)
        
        # Predict drug-cell interaction
        # Compute interaction scores for each cell-drug pair
        batch_size = cell_combined.size(0)
        drug_size = drug_combined.size(0)
        
        # Reshape for batch processing
        cell_expanded = cell_combined.unsqueeze(1).expand(batch_size, drug_size, -1)
        drug_expanded = drug_combined.unsqueeze(0).expand(batch_size, drug_size, -1)
        
        # Concatenate and predict
        pair_features = torch.cat([cell_expanded, drug_expanded], dim=2)
        pair_features_flat = pair_features.view(-1, pair_features.size(-1))
        
        # Apply predictor
        predictions = self.predictor(pair_features_flat)
        predictions = predictions.view(batch_size, drug_size)
        
        # Apply sigmoid for final predictions
        pos_adj = self.act(predictions)
        
        return pos_z, neg_z, summary_pos, summary_neg, pos_adj.view(-1)
    
    def discriminate(self, z, summary, sigmoid=True):
        # Improved discriminator with more complex scoring
        h1 = torch.matmul(z, torch.matmul(self.weight, summary))
        h2 = torch.matmul(z, torch.matmul(self.weight2, summary))
        value = h1 * torch.sigmoid(h2)  # Gating mechanism
        return torch.sigmoid(value) if sigmoid else value
    
    def loss(self, pos_z, neg_z, summary, weight_decay=1e-5):
        # Add regularization to the loss function
        pos_loss = -torch.log(self.discriminate(pos_z, summary, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(1 - self.discriminate(neg_z, summary, sigmoid=True) + EPS).mean()
        
        # L2 regularization
        l2_reg = 0.0
        for param in self.parameters():
            l2_reg += torch.norm(param, 2)
        
        return pos_loss + neg_loss + weight_decay * l2_reg
