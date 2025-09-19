import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

class EfficientMNISTModel(nn.Module):
    def __init__(self):
        super(EfficientMNISTModel, self).__init__()
        
        # Block 1: (Conv -> BN -> ReLU) x 2 -> MaxPool
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: 14x14x16
        )    

        # Block 2: (Conv -> BN -> ReLU) x 2 -> MaxPool
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: 7x7x32
        )
        
        # Global Average Pooling to prepare for the classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1) # Output: 1x1x32
        
        # Classifier Head
        self.fc = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.25), # Regularization for the classifier
            nn.Linear(64, 10), # Map features to the 10 classes
        )
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        
        # Global average pooling and flatten
        x = self.global_pool(x)
        x = x.view(x.size(0), -1) # Flatten the output for the linear layer
        
        # Classifier
        x = self.fc(x)
        
        # Return raw logits
        return x

def count_parameters(model):
    """Count the total number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_data_loaders(batch_size=128):
    """Get MNIST data loaders with augmentation"""
    
    # Training transforms with enhanced augmentation for better accuracy
    train_transform = transforms.Compose([
        transforms.RandomRotation(7),
        # transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        # transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Test transforms
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True,
                                 transform=train_transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True,
                                transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=2)
    
    return train_loader, test_loader

def train(model, device, train_loader, optimizer, scheduler, criterion, epochs):
    """Train the model for the specified number of epochs"""
    model.train()
    final_train_accuracy = 0
    final_avg_loss = 0
    
    for epoch in range(1, epochs + 1):
        correct = 0
        total = 0
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}/{epochs} | Train Batch: {batch_idx}/{len(train_loader)} '
                      f'Loss: {loss.item():.6f} '
                      f'Accuracy: {100.*correct/total:.2f}%')
        
        epoch_accuracy = 100. * correct / total
        epoch_loss = running_loss / len(train_loader)
        
        print(f'Epoch {epoch}/{epochs} completed - Accuracy: {epoch_accuracy:.2f}%, Loss: {epoch_loss:.4f}')
        
        # Step the learning rate scheduler
        scheduler.step()
        
        # Store final epoch results
        final_train_accuracy = epoch_accuracy
        final_avg_loss = epoch_loss
    
    return final_train_accuracy, final_avg_loss

def test(model, device, test_loader):
    """Test the model"""
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_accuracy, test_loss

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = EfficientMNISTModel().to(device)
    param_count = count_parameters(model)
    
    print(f"Model parameter count: {param_count:,}")
    print(f"Parameter constraint (<20,000): {'✓ PASS' if param_count < 20000 else '✗ FAIL'}")
    
    if param_count >= 20000:
        print(f"ERROR: Model has {param_count} parameters, exceeds 20,000 limit!")
        return
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(batch_size=128)
    
    # Set up optimizer and criterion
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    print("\n" + "="*50)
    print("Starting Training for 19 Epochs")
    print("="*50)
    
    start_time = time.time()
    
    # Train for exactly 1 epoch
    train_accuracy, train_loss = train(model, device, train_loader, optimizer, scheduler, criterion, epochs=19)
    
    # Test the model
    test_accuracy, test_loss = test(model, device, test_loader)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Parameters: {param_count:,} (<20,000: {'✓' if param_count < 20000 else '✗'})")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Train Accuracy: {train_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Accuracy Goal (≥99.4%): {'✓ ACHIEVED' if test_accuracy >= 99.4 else '✗ NOT ACHIEVED'}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save the model if it meets requirements
    if param_count < 20000 and test_accuracy >= 99.4:
        torch.save(model.state_dict(), 'efficient_mnist_model.pth')
        print(f"\n✓ Model saved as 'efficient_mnist_model.pth' - All requirements met!")
    
    return model, test_accuracy, param_count

if __name__ == "__main__":
    main()
