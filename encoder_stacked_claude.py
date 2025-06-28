import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, x):
        return x + self.fc(x)

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            ResidualBlock(1024),

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            ResidualBlock(512),

            nn.Linear(512, latent_dim),
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            ResidualBlock(512),

            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            ResidualBlock(1024),

            nn.Linear(1024, input_dim)
        )

    def forward(self, x):
        return self.net(x)

def train_autoencoder(trajectories, latent_dim=64, batch_size=128, num_epochs=100, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Store original trajectory lengths for reconstruction
    trajectory_lengths = [t.shape[0] for t in trajectories]
    
    # Get input dimension from the first trajectory
    input_dim = trajectories[0].shape[1]
    
    # Vertical stack all trajectories while keeping track of boundaries
    stacked_trajectories = torch.cat(trajectories, dim=0)
    
    # Create dataset and loader
    dataset = TensorDataset(stacked_trajectories)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize models
    encoder = Encoder(input_dim, latent_dim).to(device)
    decoder = Decoder(input_dim, latent_dim).to(device)
    
    # Optimizer
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        encoder.train()
        decoder.train()
        epoch_loss = 0.0
        
        for (batch,) in loader:
            x = batch.to(device)
            
            # Forward pass
            z = encoder(x)
            x_recon = decoder(z)
            
            # Compute loss
            loss = criterion(x_recon, x)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * x.size(0)
        
        epoch_loss /= len(dataset)
        
        # Evaluation
        if epoch % 10 == 0 or epoch == num_epochs:
            encoder.eval()
            decoder.eval()
            
            with torch.no_grad():
                # Compute Euclidean distance (L2 norm) for better accuracy assessment
                val_batch = stacked_trajectories.to(device)
                val_recon = decoder(encoder(val_batch))
                l2_dist = torch.norm(val_recon - val_batch, dim=1).mean().item()
                
                # Cosine similarity for directional accuracy
                cos_sim = nn.functional.cosine_similarity(val_recon, val_batch, dim=1).mean().item()
                
                print(f"Epoch {epoch:03d} | Loss: {epoch_loss:.6f} | L2 Dist: {l2_dist:.6f} | Cosine Sim: {cos_sim:.6f}")
            
            if epoch_loss < best_loss:
                best_loss = epoch_loss
    
    # Return trained models and trajectory information
    return encoder, decoder, trajectory_lengths

def encode_trajectories(encoder, trajectories, device=None):
    """
    Encode all trajectories to lower dimension while preserving trajectory structure
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder.eval()
    trajectory_lengths = [t.shape[0] for t in trajectories]
    encoded_trajectories = []
    
    with torch.no_grad():
        for trajectory in trajectories:
            # Encode each trajectory
            encoded = encoder(trajectory.to(device))
            encoded_trajectories.append(encoded)
    
    return encoded_trajectories, trajectory_lengths

def decode_trajectories(decoder, encoded_trajectories, device=None):
    """
    Decode all encoded trajectories back to original dimension
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    decoder.eval()
    decoded_trajectories = []
    
    with torch.no_grad():
        for encoded in encoded_trajectories:
            # Decode each trajectory
            decoded = decoder(encoded.to(device))
            decoded_trajectories.append(decoded)
    
    return decoded_trajectories

def evaluate_reconstruction(original_trajectories, decoded_trajectories):
    """
    Evaluate reconstruction quality across all trajectories
    """
    total_l2_dist = 0
    total_cos_sim = 0
    total_points = 0
    
    for orig, recon in zip(original_trajectories, decoded_trajectories):
        # Move tensors to the same device if needed
        if orig.device != recon.device:
            recon = recon.to(orig.device)
        
        # L2 distance
        l2_dist = torch.norm(recon - orig, dim=1).mean().item()
        
        # Cosine similarity
        cos_sim = nn.functional.cosine_similarity(recon, orig, dim=1).mean().item()
        
        # Weight by number of points in trajectory
        n_points = orig.shape[0]
        total_l2_dist += l2_dist * n_points
        total_cos_sim += cos_sim * n_points
        total_points += n_points
    
    avg_l2_dist = total_l2_dist / total_points
    avg_cos_sim = total_cos_sim / total_points
    
    return {
        "avg_l2_distance": avg_l2_dist,
        "avg_cosine_similarity": avg_cos_sim
    }

# Example usage:
if __name__ == "__main__":
    # Hyperparameters
    latent_dim = 64  # Target reduced dimension
    batch_size = 128
    lr = 1e-3
    num_epochs = 100
    
    # Assuming trajectories is a list of tensors where each tensor has shape [seq_len, 1536]
    # and seq_len can vary between trajectories
    
    # Train autoencoder
    encoder, decoder, trajectory_lengths = train_autoencoder(
        trajectories=trajectories,
        latent_dim=latent_dim,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr
    )
    
    # Encode all trajectories while preserving structure
    encoded_trajectories, _ = encode_trajectories(encoder, trajectories)
    
    # Decode all trajectories back to original dimension
    decoded_trajectories = decode_trajectories(decoder, encoded_trajectories)
    
    # Evaluate reconstruction quality
    metrics = evaluate_reconstruction(trajectories, decoded_trajectories)
    
    print("\nFinal Evaluation:")
    print(f"Average L2 Distance: {metrics['avg_l2_distance']:.6f}")
    print(f"Average Cosine Similarity: {metrics['avg_cosine_similarity']:.6f}")