# Let's load the image and generate an overlapping RGB histogram as per the request.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import pandas as pd

# Load the image
image_path = 'D:/Project Caldermeade/NJR/Training/BG/09_01_2022/DCIM/102FPLAN_14/BG_PNG/DJI_2956.png'
img = Image.open(image_path)

# HIST PLOT
# Convert image to NumPy array
# img_np = np.array(img)

# # Separate RGB channels
# r_channel = img_np[:, :, 0].flatten()
# g_channel = img_np[:, :, 1].flatten()
# b_channel = img_np[:, :, 2].flatten()

# # Create overlapping histogram for all RGB channels with log scale on y-axis
# plt.figure(figsize=(10, 7))

# plt.hist(r_channel, bins=512, color='red', alpha=0.5, label='Red Channel')
# plt.hist(g_channel, bins=512, color='green', alpha=0.5, label='Green Channel')
# plt.hist(b_channel, bins=512, color='blue', alpha=0.5, label='Blue Channel')

# plt.yscale('log')  # Apply logarithmic scale on the Y-axis
# plt.title('Overlapping RGB Histogram with Logarithmic Scale')
# plt.xlabel('Pixel Intensity Value')
# plt.ylabel('Pixel Count (Log Scale)')
# plt.legend()

# # Show the plot
# plt.tight_layout()
# plt.show()

# Var plot
#  Convert image to NumPy array

# Convert image to NumPy array
img_np = np.array(img)

# Separate RGB channels
r_channel = img_np[:, :, 0].flatten()
g_channel = img_np[:, :, 1].flatten()
b_channel = img_np[:, :, 2].flatten()

# Calculate variance across the RGB channels for each pixel
pixel_variances = np.var(img_np, axis=2).flatten()

# Create a pandas DataFrame to store the values
data = {
    'Red Channel': r_channel,
    'Green Channel': g_channel,
    'Blue Channel': b_channel,
    'Pixel Variance': pixel_variances
}

df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
csv_path = 'pixel_histogram_and_variance.csv'
df.to_csv(csv_path, index=False)

print(f"Data successfully saved to {csv_path}")

# HISTOGRAM PLOTS
# Plot overlapping RGB histograms
plt.figure(figsize=(10, 7))

plt.hist(r_channel, bins=512, color='red', alpha=0.5, label='Red Channel')
plt.hist(g_channel, bins=512, color='green', alpha=0.5, label='Green Channel')
plt.hist(b_channel, bins=512, color='blue', alpha=0.5, label='Blue Channel')

plt.yscale('log')  # Apply logarithmic scale on the Y-axis
plt.title('Overlapping RGB Histogram with Logarithmic Scale')
plt.xlabel('Pixel Intensity Value')
plt.ylabel('Pixel Count (Log Scale)')
plt.legend()

# Show the RGB histogram plot
plt.tight_layout()
plt.show()

# Plot pixel variance histogram
plt.figure(figsize=(10, 7))

plt.hist(pixel_variances, bins=512, color='purple', alpha=0.7)
plt.title('Pixel Variance Histogram')
plt.xlabel('Variance')
plt.ylabel('Pixel Count')
plt.yscale('log')  # Apply logarithmic scale to Y-axis for better visualization
plt.tight_layout()

# Show the pixel variance histogram
plt.show()

# SCATTER PLOT

# img_np = np.array(img)

# # Flatten the image array to list all pixel RGB values
# pixels = img_np.reshape(-1, 3)

# # Extract RGB components
# r = pixels[:, 0]
# g = pixels[:, 1]
# b = pixels[:, 2]

# # Create a 3D figure
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# # Plot the RGB values in 3D space
# ax.scatter(r, g, b, c=pixels / 255.0, marker='o', alpha=0.6)

# # Set axis labels
# ax.set_xlabel('Red Channel')
# ax.set_ylabel('Green Channel')
# ax.set_zlabel('Blue Channel')
# ax.set_title('3D RGB Histogram')

# # Show the plot
# plt.tight_layout()
# plt.show()