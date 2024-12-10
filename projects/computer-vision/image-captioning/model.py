import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

## Here is a fancy explanation of the Attention Module:

# Inputs to the Attention Module:
# 
#   Image features (features): This is a tensor with shape [batch, 49, embed_size], where 49 represents the 7x7 spatial 
#                              grid from the ResNet feature map, and embed_size is the dimensionality of each feature vector for a region.
#
#   Decoder hidden state (hidden): This is the hidden state from the previous GRU step in the decoder, with shape 
#                                [batch, hidden_size]. It contains information about the caption generated so far and is
#                                used to determine which regions of the image to focus on next.
#
# Repeat and Concatenate Hidden State with Image Features:
# 
#   To calculate attention scores, the hidden state is repeated across the 49 spatial positions so that it can be 
#   concatenated with each image feature.
#
#   This results in a combined tensor, combined, with shape [batch, 49, embed_size + hidden_size].
#
# Compute Attention "Energy":
# 
#   The attention module passes combined through a linear layer followed by a tanh activation to get an 
#   "energy" score for each spatial position. This energy score measures how relevant each image region is for 
#   the current decoding step, considering both the image feature and the decoder’s hidden state.
#
#   After applying the linear layer and tanh, the result energy has shape [batch, 49, hidden_size].
#
# Compute Attention Weights (Softmax):
# 
#   Next, the energy scores are passed through another linear layer (self.v) to produce attention scores for each 
#   spatial position, with shape [batch, 49, 1].
#
#   These scores are then passed through a softmax function along the spatial dimension, resulting in attention weights
#   that sum to 1 across the 49 regions. These weights indicate the importance of each region in generating the current
#   word in the caption.
#
# Weighted Sum of Image Features (Context Vector):
# 
#   The model computes the context vector by taking a weighted sum of the image features, where each feature is 
#   weighted by its corresponding attention weight.
#
#   This context vector, with shape [batch, hidden_size], represents the most relevant information from the image for
#   the current decoding step and is used as input to the GRU in the decoder.
#
# Summary: What the Attention Module Achieves
#   The attention mechanism dynamically calculates which parts of the image the decoder should focus on for each 
#   word in the caption.
#
#   It produces a context vector that represents a weighted combination of the image features, focusing on the most 
#   relevant regions based on the decoder’s current hidden state.
#
#   This context vector is then fed into the GRU along with the next word in the caption, helping the model generate
#   a word that’s both contextually and visually relevant.
#
#   By allowing the model to "attend" to different regions of the image for each word, attention improves
#   the relevance and accuracy of generated captions, especially for images with complex or detailed scenes.

class Attention(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super().__init__()
        self.attn = nn.Linear(embed_size + hidden_size, hidden_size)  # Transform to hidden_size for GRU compatibility
        self.v = nn.Linear(hidden_size, 1, bias=False)  # Attention score

    def forward(self, features, hidden):
        # features: [batch, 49, embed_size]
        # hidden: [batch, hidden_size]

        # Repeat hidden state to match features shape and to enable a comparison of each spatial position with the hidden state
        hidden = hidden.unsqueeze(1).repeat(1, features.size(1), 1)  # [batch, 49, hidden_size]

        # Calculate attention scores
        combined = torch.cat((features, hidden), dim=2)  # [batch, 49, embed_size + hidden_size]
        energy = torch.tanh(self.attn(combined))  # [batch, 49, hidden_size]
        attention = F.softmax(self.v(energy), dim=1)  # [batch, 49, 1]

        # Apply attention weights to the features
        context = (attention * features).sum(dim=1)  # Weighted sum, now [batch, hidden_size]
        return context, attention

class DecoderGRU(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(embed_size, hidden_size)
        
        # for embed_size * 2 to work both embed_sizes (of the feature vectors and the word vectors) must be the same
        # otherwise I could pass the image feature vector size as a parameter
        self.gru = nn.GRU(embed_size * 2, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, features, captions):
        # Embed captions
        captions = self.dropout(self.embed(captions[:, :-1]))  # Ignore last token during training
        batch_size = features.size(0)
        seq_len = captions.size(1)

        # Initialize hidden state as zeros
        hidden = torch.zeros(1, batch_size, self.gru.hidden_size).to(features.device)

        # Store outputs for each time step
        outputs = torch.zeros(batch_size, seq_len, self.fc.out_features).to(features.device)

        # Process each word in the caption sequence
        for t in range(seq_len):
            context, _ = self.attention(features, hidden[-1])  # Apply attention
            # context shape is [batch, embed_size]
            # captions[:, t, :] shape is [batch, embed_size]
            gru_input = torch.cat((captions[:, t, :], context), dim=1).unsqueeze(1)  # [batch, 1, embed_size + embed_size]
            #print(f"gru_input: {gru_input.shape}, context: {context.shape}, hidden: {hidden.shape}, outputs: {outputs.shape}")
            output, hidden = self.gru(gru_input, hidden)  # GRU step
            outputs[:, t, :] = self.fc(output.squeeze(1))  # Output for time step t

        return outputs

    def sample(self, features, max_len=20):
        """Generate a caption using greedy search."""
        sampled_ids = []
        batch_size = features.size(0)
        hidden = torch.zeros(1, batch_size, self.gru.hidden_size).to(features.device)
        inputs = torch.zeros(batch_size, dtype=torch.long).to(features.device)  # Assumes <start> token is index 0

        for _ in range(max_len):
            embedded_inputs = self.embed(inputs).unsqueeze(1)  # [batch, 1, embed_size]
            context, _ = self.attention(features, hidden[-1])  # [batch, embed_size]
            gru_input = torch.cat((embedded_inputs, context.unsqueeze(1)), dim=2)  # [batch, 1, embed_size + hidden_size]
            output, hidden = self.gru(gru_input, hidden)
            output = self.fc(output.squeeze(1))  # [batch, vocab_size]
            _, predicted = output.max(1)  # Get highest scoring word
            sampled_ids.append(predicted)
            inputs = predicted  # Set input as predicted word for next step

            # Stop if all batches predicted <end> token (assuming <end> token is 1)
            if (predicted == 1).all():
                break

        sampled_ids = torch.stack(sampled_ids, dim=1)  # Convert list to tensor
        return sampled_ids
    
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        # Removing the average pooling and fully connected layer to keep more spatial information for the sequence approach
        modules = list(resnet.children())[:-2]  
        self.resnet = nn.Sequential(*modules)

        # Map the ResNet feature map to the desired embedding size
        self.embed = nn.Conv2d(2048, embed_size, kernel_size=1)  # 2048 channels in ResNet50's final conv layer

    def forward(self, images):
        # Extract the feature map from the ResNet
        features = self.resnet(images)  # Shape: [batch, 2048, 7, 7] for typical ResNet50 output
        features = self.embed(features)  # Shape: [batch, embed_size, 7, 7]

        # Reshape feature map into a sequence from [batch, embed_size, 7, 7] and embed_size is here channels
        # 0, 2, 3, 1 are the indexes of the feature map before we permutate the axes
        features = features.permute(0, 2, 3, 1)  # Shape: [batch, 7, 7, embed_size]
        # we keep the batch size (size(0)), let dim 2 and 3 infer (7x7=49 features), and set the last dimension to the embed size
        features = features.view(features.size(0), -1, features.size(-1))  # Shape: [batch, 49, embed_size]

        return features

class BaseLineDecoderCNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        # Embed the captions (ignore the last caption token for teacher forcing during training)
        captions = self.embed(captions[:, :-1])  # Shape: [batch, seq_len, embed_size]

        # Concatenate image features and caption embeddings
        inputs = torch.cat((features, captions), dim=1)  # Shape: [batch, 49 + seq_len, embed_size]

        # Pass through LSTM
        lstm_hiddens, _ = self.lstm(inputs)  # Shape: [batch, 49 + seq_len, hidden_size]

        # Pass LSTM output through the fully connected layer
        outputs = self.fc(lstm_hiddens)  # Shape: [batch, 49 + seq_len, vocab_size]

        return outputs

    def sample(self, inputs, states=None, max_len=20):
        """Generates a caption for a given image feature vector"""
        sampled_ids = []
        for _ in range(max_len):
            # Pass through LSTM cell
            lstm_out, states = self.lstm(inputs, states)  # Shape: [batch, 1, hidden_size]
            outputs = self.fc(lstm_out.squeeze(1))  # Shape: [batch, vocab_size]
            _, predicted = outputs.max(1)  # Get the index of the max log-probability
            sampled_ids.append(predicted)

            # Stop generation if `<end>` token is predicted
            if predicted.item() == 1:  # Assuming `1` is the index of `<end>` see Preliminiaries.ipynb
                break

            # Embed the predicted word and use as next input
            inputs = self.embed(predicted).unsqueeze(1)  # Shape: [batch, 1, embed_size]

        sampled_ids = torch.stack(sampled_ids, dim=1)  # Convert list to tensor
        return sampled_ids
