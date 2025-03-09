import torch
import time
import argparse
import numpy as np
from model import *
from data_process import process
from my_utiils import metrics_graph
from data_load import dataload
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import the improved classes from the previous code
# Note: These imports would come from your local file, but I'm including them here
# from improved_model import ImprovedGraphCDR, GATv2Encoder, AttentionSummary, NodeRepresentation
# Instead, we're assuming you've updated your model.py file with the improved implementation

parser = argparse.ArgumentParser(description='Drug_response_prediction_with_improved_GAT')
parser.add_argument('--alph', dest='alph', type=float, default=0.25, help='Weight for positive DGI loss')
parser.add_argument('--beta', dest='beta', type=float, default=0.25, help='Weight for negative DGI loss')
parser.add_argument('--epoch', dest='epoch', type=int, default=500, help='Number of training epochs')
parser.add_argument('--hidden_channels', dest='hidden_channels', type=int, default=256, help='Hidden dimension size')
parser.add_argument('--output_channels', dest='output_channels', type=int, default=100, help='Output dimension size')
parser.add_argument('--lr', dest='lr', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-5, help='Weight decay for regularization')
parser.add_argument('--dropout', dest='dropout', type=float, default=0.2, help='Dropout rate')
parser.add_argument('--num_heads', dest='num_heads', type=int, default=4, help='Number of attention heads')
parser.add_argument('--patience', dest='patience', type=int, default=20, help='Patience for early stopping')
parser.add_argument('--seed', dest='seed', type=int, default=42, help='Random seed')
args = parser.parse_args()

# Set random seeds for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

start_time = time.time()
print(f"Using device: {device}")
print("Starting data loading and processing...")

# Data files
Drug_info_file = '../data/Drug/1.Drug_listMon Jun 24 09_00_55 2019.csv'
IC50_threds_file = '../data/Drug/drug_threshold.txt'
Drug_feature_file = '../data/Drug/drug_graph_feat'
Cell_line_info_file = '../data/Celline/Cell_lines_annotations.txt'
Genomic_mutation_file = '../data/Celline/genomic_mutation_34673_demap_features.csv'
Cancer_response_exp_file = '../data/Celline/GDSC_IC50.csv'
Gene_expression_file = '../data/Celline/genomic_expression_561celllines_697genes_demap_features.csv'
Methylation_file = '../data/Celline/genomic_methylation_561celllines_808genes_demap_features.csv'

# Bio-feature extraction
drug_feature, mutation_feature, gexpr_feature, methylation_feature, data_new, nb_celllines, nb_drugs = dataload(
    Drug_info_file, IC50_threds_file, Drug_feature_file, Cell_line_info_file, Genomic_mutation_file,
    Cancer_response_exp_file, Gene_expression_file, Methylation_file)

# Split train and test sets 
drug_set, cellline_set, train_edge, label_pos, train_mask, test_mask, atom_shape = process(
    drug_feature, mutation_feature, gexpr_feature, methylation_feature, data_new, nb_celllines, nb_drugs)

print(f"Data loaded. Number of cell lines: {nb_celllines}, Number of drugs: {nb_drugs}")
print(f"Input feature dimensions - Drugs: {atom_shape}, Gene expression: {gexpr_feature.shape[-1]}, Methylation: {methylation_feature.shape[-1]}")

# Move data to device if using GPU
if torch.cuda.is_available():
    drug_set = [d.to(device) for d in drug_set]
    cellline_set = [(c[0].to(device), c[1].to(device), c[2].to(device)) for c in cellline_set]
    train_edge = torch.tensor(train_edge, device=device)
    label_pos = label_pos.to(device)
    train_mask = train_mask.to(device)
    test_mask = test_mask.to(device)

# Create improved node representation module with the specified hyperparameters
node_representation = NodeRepresentation(
    gcn_layer=atom_shape,
    dim_gexp=gexpr_feature.shape[-1],
    dim_methy=methylation_feature.shape[-1],
    output=args.output_channels,
    units_list=[args.hidden_channels, args.hidden_channels, args.hidden_channels],
    dropout=args.dropout,
    num_heads=args.num_heads
)

# Create improved encoder
encoder = GATv2Encoder(
    in_channels=args.output_channels, 
    hidden_channels=args.hidden_channels,
    num_heads=args.num_heads,
    dropout=args.dropout
)

# Create improved summary module
summary = AttentionSummary(
    ino=args.output_channels, 
    inn=args.hidden_channels
)

# Create the improved GraphCDR model
model = ImprovedGraphCDR(
    hidden_channels=args.hidden_channels,
    encoder=encoder,
    summary=summary,
    feat=node_representation,
    index=nb_celllines,
    dropout=args.dropout
)

# Move model to device
model = model.to(device)
print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

# Optimizer with weight decay for L2 regularization
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=args.lr, 
    weight_decay=args.weight_decay,
    betas=(0.9, 0.999)
)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(
    optimizer, 
    mode='max', 
    factor=0.5, 
    patience=10, 
    verbose=True, 
    min_lr=1e-6
)

# Loss function
myloss = nn.BCELoss()

# Gradient clipping function
def clip_gradients(model, clip_value=1.0):
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

def train():
    model.train()
    loss_temp = 0
    
    for batch, (drug, cell) in enumerate(zip(drug_set, cellline_set)):
        optimizer.zero_grad()
        
        # Forward pass
        pos_z, neg_z, summary_pos, summary_neg, pos_adj = model(
            drug.x, drug.edge_index, drug.batch, cell[0], cell[1], cell[2], train_edge)
        
        # Calculate losses
        dgi_pos = model.loss(pos_z, neg_z, summary_pos)
        dgi_neg = model.loss(neg_z, pos_z, summary_neg)
        pos_loss = myloss(pos_adj[train_mask], label_pos[train_mask])
        
        # Combined loss with weights
        loss = (1 - args.alph - args.beta) * pos_loss + args.alph * dgi_pos + args.beta * dgi_neg
        
        # Backward pass
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        clip_gradients(model, clip_value=1.0)
        
        # Optimizer step
        optimizer.step()
        
        loss_temp += loss.item()
    
    avg_loss = loss_temp / len(drug_set)
    print(f'Train loss: {avg_loss:.4f}')
    return avg_loss

def validate():
    model.eval()
    with torch.no_grad():
        for batch, (drug, cell) in enumerate(zip(drug_set, cellline_set)):
            _, _, _, _, pre_adj = model(drug.x, drug.edge_index, drug.batch, cell[0], cell[1], cell[2], train_edge)
            val_loss = myloss(pre_adj[test_mask], label_pos[test_mask])
        
        yp = pre_adj[test_mask].cpu().numpy()
        ytest = label_pos[test_mask].cpu().numpy()
        
        AUC, AUPR, F1, ACC = metrics_graph(ytest, yp)
        print(f'Validation loss: {val_loss.item():.4f}')
        print(f'Validation AUC: {AUC:.4f}, AUPR: {AUPR:.4f}, F1: {F1:.4f}, ACC: {ACC:.4f}')
    
    return AUC, AUPR, F1, ACC, val_loss.item()

# Early stopping
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_auc, model):
        if self.best_score is None:
            self.best_score = val_auc
            return True
        
        if val_auc > self.best_score + self.min_delta:
            self.best_score = val_auc
            self.counter = 0
            return True
        
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return False

# Initialize early stopping
early_stopping = EarlyStopping(patience=args.patience)

# Training loop with validation
best_model_state = None
best_epoch = 0
train_losses = []
val_metrics = {'auc': [], 'aupr': [], 'f1': [], 'acc': [], 'loss': []}

print("Starting training...")
for epoch in range(args.epoch):
    print(f'\nEpoch: {epoch + 1}/{args.epoch}')
    
    # Train for one epoch
    train_loss = train()
    train_losses.append(train_loss)
    
    # Evaluate on validation set
    auc, aupr, f1, acc, val_loss = validate()
    val_metrics['auc'].append(auc)
    val_metrics['aupr'].append(aupr)
    val_metrics['f1'].append(f1)
    val_metrics['acc'].append(acc)
    val_metrics['loss'].append(val_loss)
    
    # Update learning rate based on validation performance
    scheduler.step(auc)
    
    # Check if this is the best model so far
    is_best = early_stopping(auc, model)
    if is_best:
        best_model_state = model.state_dict().copy()
        best_epoch = epoch
        print(f"New best model saved at epoch {epoch + 1}")
    
    # Check early stopping
    if early_stopping.early_stop:
        print(f"Early stopping triggered at epoch {epoch + 1}")
        break

# Load the best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"Loaded best model from epoch {best_epoch + 1}")

# Final evaluation
print("\nFinal Evaluation:")
final_auc, final_aupr, final_f1, final_acc, _ = validate()

# Save the model
torch.save({
    'epoch': best_epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'auc': final_auc,
    'aupr': final_aupr,
    'f1': final_f1,
    'acc': final_acc,
}, 'best_graphcdr_model.pth')

# Calculate elapsed time
elapsed = time.time() - start_time
print('\n---------------------------------------')
print(f'Total training time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)')
print(f'Best epoch: {best_epoch + 1}')
print(f'Final AUC: {final_auc:.4f}, AUPR: {final_aupr:.4f}, F1: {final_f1:.4f}, ACC: {final_acc:.4f}')
print('---------------------------------------')

# Print best hyperparameters for reference
print("\nModel Hyperparameters:")
print(f"Learning rate: {args.lr}")
print(f"Hidden channels: {args.hidden_channels}")
print(f"Output channels: {args.output_channels}")
print(f"Dropout: {args.dropout}")
print(f"Number of attention heads: {args.num_heads}")
print(f"Alpha (DGI pos weight): {args.alph}")
print(f"Beta (DGI neg weight): {args.beta}")
print(f"Weight decay: {args.weight_decay}")
print('---------------------------------------')
