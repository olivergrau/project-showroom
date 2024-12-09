import torch

class ImageCaptioningPredictor:
    def __init__(self, encoder, decoder, vocab, device='cpu'):
        """
        Initialize the Image Captioning Model with the trained encoder and decoder.

        :param encoder: The encoder model (e.g., EncoderCNN)
        :param decoder: The decoder model (e.g., DecoderGRU with attention)
        :param vocab: The vocabulary object (should contain word-to-index and index-to-word mappings)
        :param device: The device to run inference on ('cpu' or 'cuda')
        """
        self.device = device
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.vocab = vocab

        # Set models to evaluation mode
        self.encoder.eval()
        self.decoder.eval()

    def generate_caption(self, img_tensor, max_length=20):
        """
        Generate a caption for the input image using the decoder's sample method.
    
        :param img_tensor: Preprocessed image tensor
        :param max_length: Maximum length of the generated caption
        :return: Generated caption as a string
        """
        
        #print(f"shape of img_tensor: {img_tensor.shape}")
        # Ensure the image is on the correct device
        img_tensor = img_tensor.to(self.device)
    
        # Extract features from the image
        with torch.no_grad():
            features = self.encoder(img_tensor)  # Shape: [1, num_regions, embed_size]
        
        #print(f"features shape: {features.shape}")
        # Use the decoder's sample method to generate the caption
        caption_indices = self.decoder.sample(features, max_len=max_length)  # Shape: [1, seq_len]
    
        # Convert the word indices to words and join to form the final caption
        caption = [
            self.vocab.idx2word[idx.item()]  # Convert tensor to integer
            for idx in caption_indices.squeeze()  # Squeeze to 1D tensor
            if idx.item() != self.vocab('<end>')  # Convert idx to integer before comparison
        ]
    
        return ' '.join(caption)
