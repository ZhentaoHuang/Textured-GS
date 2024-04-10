import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import torchvision.transforms as transforms



def read_image_to_tensor(image_path):
    """
    Reads an image from the specified path and converts it to a PyTorch tensor.
    
    Parameters:
        image_path (str): Path to the image file.
        
    Returns:
        torch.Tensor: A float tensor of shape (C, H, W) in the range [0, 1].
    """
    # Read the image
    image = Image.open(image_path)
    
    # Define the transformation: convert the image to tensor and normalize it to [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),  # This converts the image to a tensor and scales to [0, 1]
    ])
    
    # Apply the transformation
    image_tensor = transform(image)
    
    # If the image is grayscale, convert it to RGB by repeating the channels
    if image_tensor.shape[0] == 1:
        image_tensor = image_tensor.repeat(3, 1, 1)
    
    return image_tensor



class EnhancedTextureFunction(nn.Module):
    def __init__(self, num_components=10, image_size=(256, 256)):
        super(EnhancedTextureFunction, self).__init__()
        self.num_components = num_components
        self.image_size = image_size

        # Frequencies for sine and cosine components
        self.frequencies = nn.Parameter(torch.randn(num_components, 2))  # [num_components, 2] for X and Y

        # Amplitudes and phase shifts for components
        self.amplitudes = nn.Parameter(torch.rand(num_components))  # Amplitude for each component
        self.phases = nn.Parameter(torch.rand(num_components))  # Phase shift for each component

        # Overall phase shift for RGB channels to allow color variation
        self.rgb_phases = nn.Parameter(torch.rand(3))  # [R, G, B]

    def forward(self):
        H, W = self.image_size
        grid_y, grid_x = torch.meshgrid(torch.linspace(0, 1, steps=H), torch.linspace(0, 1, steps=W), indexing='ij')

        # Initialize texture as zero
        texture = torch.zeros(H, W, 3, dtype=torch.float32, device=self.frequencies.device)

        # Add contributions from each component
        for i in range(self.num_components):
            for c in range(3):  # RGB channels
                # Compute spatial frequency component
                spatial_frequency = (self.frequencies[i, 0] * grid_x + self.frequencies[i, 1] * grid_y)
                # Sum of sinusoids model
                texture[:, :, c] += self.amplitudes[i] * torch.sin(2 * np.pi * spatial_frequency + self.phases[i] + self.rgb_phases[c])

        # Normalize texture to [0, 1]
        texture = (texture - texture.min()) / (texture.max() - texture.min())
        texture = torch.sigmoid(texture)
        return texture




if __name__ == "__main__":

    target_image = read_image_to_tensor("000001.jpg")
    print("target", target_image.shape)

    target_texture = torch.tensor(target_image, dtype=torch.float32).unsqueeze(0) # Normalize

    y_start, x_start = 480, 700  # Example start position
    h, w = 50, 50  # Example size of the patch

    # Extract the patch
    target_texture = target_texture[:, :, y_start:y_start+h, x_start:x_start+w]

    criterion = nn.MSELoss()
    model = EnhancedTextureFunction(num_components=10, image_size=(h, w))
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

    x = torch.linspace(0, np.pi, w, dtype=torch.float32)
    y = torch.linspace(0, np.pi, h, dtype=torch.float32)
    X, Y = torch.meshgrid(x, y)

    initial_output = model()
    print("initial_out", initial_output.shape)
    initial_output = initial_output.permute(2, 0, 1)  # Reorder dimensions to [Channels, Height, Width]
    initial_output = initial_output.unsqueeze(0)  # Add a batch dimension, making it [1, Channels, Height, Width]

    for epoch in range(2000): # Number of epochs
        optimizer.zero_grad()
        output = model()

        output = output.permute(2, 0, 1)  # Reorder dimensions to [Channels, Height, Width]
        output = output.unsqueeze(0)  # Add a batch dimension, making it [1, Channels, Height, Width]

        # print("shape:", output.shape, target_texture.shape)
        loss = criterion(output, target_texture)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')


    final_output = model()
    final_output = final_output.permute(2, 0, 1)  # Reorder dimensions to [Channels, Height, Width]
    final_output = final_output.unsqueeze(0)  # Add a batch dimension, making it [1, Channels, Height, Width]


    # Assuming 'output_tensor' is your model's final output in the shape [1, Channels, Height, Width]
    # Convert to numpy array and scale if necessary
    output_image = final_output.squeeze().permute(1, 2, 0).detach().numpy()
    output_image = np.clip(output_image, 0, 1)  # Ensure the image's values are in the range [0, 1]
    # print("Output tensor min:", final_output.min().item(), "max:", final_output.max().item())

    # Do the same for the target texture if it's not already in the correct format for visualization
    target_image = target_texture.squeeze().permute(1, 2, 0).detach().numpy()
    target_image = np.clip(target_image, 0, 1)  # Ensure the image's values are in the range [0, 1]

    # Do the same for the target texture if it's not already in the correct format for visualization
    initial_image = initial_output.squeeze().permute(1, 2, 0).detach().numpy()
    initial_image = np.clip(initial_image, 0, 1)  # Ensure the image's values are in the range [0, 1]

    # Create a plot to show the output and target side by side
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))

    # Display the Initial texture
    ax[0].imshow(initial_image)
    ax[0].set_title("InitialTexture")
    ax[0].axis('off')  # Hide the axes

    # Display the learned texture
    ax[1].imshow(output_image)
    ax[1].set_title("Learned Texture")
    ax[1].axis('off')  # Hide the axes

    # Display the target texture
    ax[2].imshow(target_image)
    ax[2].set_title("Target Texture")
    ax[2].axis('off')  # Hide the axes

    plt.show()

