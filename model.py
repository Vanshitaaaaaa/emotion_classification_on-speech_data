import torch
import torch.nn as nn
import torch.nn.functional as F

class StatisticalPooling(nn.Module):
    """Statistical pooling unit for temporal feature aggregation"""
    def forward(self, x):
        # x shape: [batch, time, features]
        mean = torch.mean(x, dim=1)
        std = torch.std(x, dim=1)
        return torch.cat([mean, std], dim=1)

class MultiscaleConvBlock(nn.Module):
    """Encapsulates multi-scale convolution with residual connection"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        c1 = out_channels // 3
        c2 = out_channels // 3
        c3 = out_channels - (c1 + c2)
        
        self.conv1x1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels, c2, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, c3, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        conv1 = self.conv1x1(x)
        conv3 = self.conv3x3(x)
        conv5 = self.conv5x5(x)
        
        out = torch.cat([conv1, conv3, conv5], dim=1)
        out = self.bn(out)
        residual = self.residual(x)
        
        return F.relu(out + residual)

class EnhancedCNNLSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # Multi-scale CNN blocks
        self.conv_block1 = MultiscaleConvBlock(1, 32)
        self.conv_block2 = MultiscaleConvBlock(32, 64)
        
        # Pooling layers
        self.pool1 = nn.MaxPool2d((1, 2))
        self.pool2 = nn.MaxPool2d((1, 3))
        
        # LSTM configuration
        self.lstm_input_size = 64 * 9
        self.lstm_hidden = 96
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=self.lstm_hidden * 2,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Statistical pooling
        self.stat_pooling = StatisticalPooling()
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(self.lstm_hidden * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        self._init_weights()

    def _init_weights(self):
        """Advanced weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.kaiming_normal_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, x):
        # CNN processing
        x = self.conv_block1(x)
        x = self.pool1(x)
        x = F.dropout(x, 0.3, self.training)
        
        x = self.conv_block2(x)
        x = self.pool2(x)
        x = F.dropout(x, 0.3, self.training)
        
        # Reshape for LSTM
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(x.size(0), x.size(1), -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        enhanced_features = lstm_out + attn_out
        
        # Statistical pooling
        pooled_features = self.stat_pooling(enhanced_features)
        
        # Classification
        return self.fc(pooled_features)

class AdaptiveDropout(nn.Module):
    """Adaptive dropout for training stability"""
    def __init__(self, initial_p=0.5, final_p=0.1, total_epochs=100):
        super().__init__()
        self.initial_p = initial_p
        self.final_p = final_p
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def forward(self, x):
        if self.training:
            current_p = self.initial_p - (self.initial_p - self.final_p) * (self.current_epoch / self.total_epochs)
            return F.dropout(x, current_p, True)
        return x

    def step_epoch(self):
        self.current_epoch = min(self.current_epoch + 1, self.total_epochs)

def create_enhanced_model(num_classes=8):
    return EnhancedCNNLSTM(num_classes)
