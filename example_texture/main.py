import torch
import torch.nn as nn
import torch.optim as optim

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


class TextureFunction(nn.Module):




    def __init__(self):
        super(TextureFunction, self).__init__()
        # Initialize parameters
        self.alpha = nn.Parameter(torch.rand(1)) # Randomly initialized
        self.beta = nn.Parameter(torch.rand(1))
        self.gamma = nn.Parameter(torch.rand(1) )
        self.phi_R = nn.Parameter(torch.rand(1) )
        self.phi_G = nn.Parameter(torch.rand(1) )
        self.phi_B = nn.Parameter(torch.rand(1) )
        self.t = nn.ParameterList([nn.Parameter(torch.rand(1)) for _ in range(3)])
        self.a_n = nn.ParameterList([nn.Parameter(torch.rand(1)) for _ in range(3)])
        self.b_n = nn.ParameterList([nn.Parameter(torch.rand(1)) for _ in range(3)])




    def forward(self, X, Y):
        # print(self.t.shape)
        R = self.a_n[0] * torch.sin(self.alpha * X) + self.b_n[0] * torch.cos(self.phi_R * Y) + self.t[0]
        G = self.a_n[1] * torch.sin(self.beta * X) + self.b_n[1] * torch.cos(self.phi_G * Y) + self.t[1]
        B = self.a_n[2] * torch.sin(self.gamma * X) + self.b_n[2] * torch.cos(self.phi_B * Y) + self.t[2]
        RGB = torch.stack([R, G, B], dim=-1)
        RGB = (RGB - RGB.min()) / (RGB.max() - RGB.min())
        return RGB



class PatternGenerator(nn.Module):
    def __init__(self):
        super(PatternGenerator, self).__init__()
        # Initialize parameters you want to learn
        self.freq_x = nn.Parameter(torch.tensor(3.0))  # Frequency for the x dimension
        self.freq_y = nn.Parameter(torch.tensor(3.0))  # Frequency for the y dimension

    def forward(self, X, Y):
        # Your pattern function adapted for PyTorch and learnable parameters
        Z = torch.sin(self.freq_x * X) + torch.cos(self.freq_y * Y) + torch.sqrt(X**2 + Y**2)
        return Z



if __name__ == "__main__":

    target_image = read_image_to_tensor("000001.jpg")
    print("target", target_image.shape)

    target_texture = torch.tensor(target_image, dtype=torch.float32).unsqueeze(0) # Normalize

    y_start, x_start = 200, 500  # Example start position
    h, w = 5, 5  # Example size of the patch

    # Extract the patch
    target_texture = target_texture[:, :, y_start:y_start+h, x_start:x_start+w]

    criterion = nn.MSELoss()
    model = TextureFunction()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    x = torch.linspace(0, np.pi, w, dtype=torch.float32)
    y = torch.linspace(0, np.pi, h, dtype=torch.float32)
    X, Y = torch.meshgrid(x, y)

    initial_output = model(X, Y)
    print("initial_out", initial_output.shape)
    initial_output = initial_output.permute(2, 0, 1)  # Reorder dimensions to [Channels, Height, Width]
    initial_output = initial_output.unsqueeze(0)  # Add a batch dimension, making it [1, Channels, Height, Width]

    for epoch in range(10000): # Number of epochs
        optimizer.zero_grad()
        output = model(X, Y)

        output = output.permute(2, 0, 1)  # Reorder dimensions to [Channels, Height, Width]
        output = output.unsqueeze(0)  # Add a batch dimension, making it [1, Channels, Height, Width]
  
        # print("shape:", output.shape, target_texture.shape)
        loss = criterion(output, target_texture)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')


    final_output = model(X, Y)
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


