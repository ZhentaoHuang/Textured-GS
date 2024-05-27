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


class RGBSphericalHarmonicsTexture(nn.Module):
    def __init__(self, image_size=(256, 256), degree=3):
        super(RGBSphericalHarmonicsTexture, self).__init__()
        self.image_size = image_size
        # Calculate the number of coefficients (sum_{l=0}^degree (2*l + 1))
        num_coeffs = sum((2 * l + 1) for l in range(degree + 1))
        self.coeffs_r = nn.Parameter(torch.randn(num_coeffs))
        self.coeffs_g = nn.Parameter(torch.randn(num_coeffs))
        self.coeffs_b = nn.Parameter(torch.randn(num_coeffs))

    def forward(self):
        H, W = self.image_size
        grid_size = 150
        # Create a grid of x, y, z coordinates
        # theta, phi = torch.meshgrid(torch.linspace(0, np.pi, steps=H), torch.linspace(0, 2 * np.pi, steps=W), indexing='ij')
        # x = torch.sin(phi) * torch.cos(theta)
        # y = torch.sin(phi) * torch.sin(theta)
        # z = torch.cos(phi)


         # Create a grid of x, y in the range [-1, 1]
        grid_x, grid_y = torch.meshgrid(torch.linspace(-0.5, 0.5, steps=grid_size), torch.linspace(-0.5, 0.5, steps=grid_size), indexing='ij')
        radius_squared = grid_x**2 + grid_y**2
        valid_mask = radius_squared <= 1.0
        grid_z = torch.zeros_like(grid_x)

        # Calculate z only for valid x, y points
        grid_z[valid_mask] = torch.sqrt(1 - radius_squared[valid_mask])

        # Normalize x and y where the radius squared is greater than 1
        invalid_mask = ~valid_mask
        lengths = torch.sqrt(grid_x[invalid_mask]**2 + grid_y[invalid_mask]**2)
        grid_x[invalid_mask] /= lengths
        grid_y[invalid_mask] /= lengths



        x = grid_x
        y = grid_y
        z = grid_z


        # Initialize texture
        texture = torch.zeros(H, W, 3, dtype=torch.float32, device=self.coeffs_r.device)

        # Define spherical harmonics using Cartesian coordinates
        P_00 = torch.full((H, W), 0.5 / np.sqrt(np.pi), dtype=torch.float32, device=self.coeffs_r.device)
        P_1m1 = np.sqrt(3/(2*np.pi)) * y
        P_10 = np.sqrt(3/(4*np.pi)) * z
        P_11 = np.sqrt(3/(2*np.pi)) * x
        P_2m2 = 0.5 * np.sqrt(15/np.pi) * (2 * x * y)
        P_2m1 = np.sqrt(15/(2*np.pi)) * y * z
        P_20 = 0.25 * np.sqrt(5/np.pi) * (2 * z**2 - x**2 - y**2)
        P_21 = np.sqrt(15/(2*np.pi)) * x * z
        P_22 = 0.5 * np.sqrt(15/np.pi) * (x**2 - y**2)
        P_3m3 = np.sqrt(35/(2*np.pi)) * (y * (3 * x**2 - y**2))
        P_3m2 = np.sqrt(105/np.pi) * (x * y * z)
        P_3m1 = np.sqrt(21/(4*np.pi)) * y * (5 * z**2 - 1)
        P_30 = 0.25 * np.sqrt(7/np.pi) * z * (5 * z**2 - 3)
        P_31 = np.sqrt(21/(4*np.pi)) * x * (5 * z**2 - 1)
        P_32 = np.sqrt(105/np.pi) * z * (x**2 - y**2)
        P_33 = np.sqrt(35/(2*np.pi)) * (x * (x**2 - 3 * y**2))

        harmonics = torch.stack([P_00, P_1m1, P_10, P_11, P_2m2, P_2m1, P_20, P_21, P_22, P_3m3, P_3m2, P_3m1, P_30, P_31, P_32, P_33], dim=0)
        # print(self.coeffs_r.shape)
        # Compute texture for each channel using spherical harmonics coefficients
        for i in range(16):  # Ensure correct indexing according to the number of coefficients
            texture[:, :, 0] += self.coeffs_r[i] * harmonics[i, :, :]
            texture[:, :, 1] += self.coeffs_g[i] * harmonics[i, :, :]
            texture[:, :, 2] += self.coeffs_b[i] * harmonics[i, :, :]


        # Normalize texture to [0, 1]
        # texture = (texture - texture.min()) / (texture.max() - texture.min())
        texture = torch.sigmoid(texture)  # Apply sigmoid to clamp the values between 0 and 1

        return texture



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
        # texture = (texture - texture.min()) / (texture.max() - texture.min())
        # texture = torch.sigmoid(texture)
        return texture




if __name__ == "__main__":

    target_image = read_image_to_tensor("00051.png")
    print("target", target_image.shape)

    target_texture = torch.tensor(target_image, dtype=torch.float32).unsqueeze(0) # Normalize

    y_start, x_start = 203, 206  # Example start position
    h, w = 150, 150  # Example size of the patch

    # Extract the patch
    target_texture = target_texture[:, :, y_start:y_start+h, x_start:x_start+w]

    criterion = nn.MSELoss()
    model = RGBSphericalHarmonicsTexture(image_size=(h, w))
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

    x = torch.linspace(0, np.pi, w, dtype=torch.float32)
    y = torch.linspace(0, np.pi, h, dtype=torch.float32)
    X, Y = torch.meshgrid(x, y)

    initial_output = model()
    print("initial_out", initial_output.shape)
    initial_output = initial_output.permute(2, 0, 1)  # Reorder dimensions to [Channels, Height, Width]
    initial_output = initial_output.unsqueeze(0)  # Add a batch dimension, making it [1, Channels, Height, Width]

    for epoch in range(10000): # Number of epochs
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

