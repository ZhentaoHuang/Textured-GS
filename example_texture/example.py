import numpy as np
import matplotlib.pyplot as plt


def generate_pattern(width, height):
    x = np.linspace(-np.pi, np.pi, width)
    y = np.linspace(-np.pi, np.pi, height)
    X, Y = np.meshgrid(x, y)
    
    # Combining a trigonometric function with a radial gradient
    Z = np.sin(3 * X) + np.cos(3 * Y) + np.sqrt(X**2 + Y**2)
    Z = (Z - Z.min()) / (Z.max() - Z.min())  # Normalize to [0, 1]
    
    return Z



def generate_texture(width, height, alpha, beta, gamma, phi_R, phi_G, phi_B):
    x = np.linspace(0, np.pi, width, endpoint=False)
    y = np.linspace(0, np.pi, height, endpoint=False)
    X, Y = np.meshgrid(x, y) # Create a 2D grid of x, y coordinates

    # Calculate R, G, B components using the texture function
    R = 128.5 + 127.5 * np.sin(alpha * X + phi_R)
    G = 128.5 + 127.5 * np.sin(beta * Y + phi_G)
    B = 128.5 + 127.5 * np.sin(gamma * (X + Y) + phi_B)

    # Stack R, G, B components along a new axis to create an image
    texture = np.stack([R, G, B], axis=-1).astype(np.uint8)
    return texture

# # Parameters for the texture function
# alpha, beta, gamma = 5, 8, 3
# phi_R, phi_G, phi_B = 0, 2, 4

# # Generate and visualize the texture
# texture = generate_texture(256, 256, alpha, beta, gamma, phi_R, phi_G, phi_B)
# plt.imshow(texture)
# plt.axis('off') # Hide axis
# plt.show()


pattern = generate_pattern(512, 512)
plt.imshow(pattern, cmap='viridis')
plt.axis('off')
plt.show()