from PIL import Image
import numpy as np
#from scipy.fftpack import dct, idct

def mdct(x):
    """Compute the MDCT of the 1D input array x."""
    N = len(x)
    n0 = (N + 1) / 2
    X = np.zeros(N//2)

    for k in range(N//2):
        X[k] = np.sum([
            x[n] * np.cos(np.pi / N * (n + n0) * (2 * k + 1))
            for n in range(N)
        ])
    
    return X

def imdct(X):
    """Compute the IMDCT of the 1D input array X."""
    N = 2 * len(X)
    n0 = (N + 1) / 2
    x = np.zeros(N)

    for n in range(N):
        x[n] = np.sum([
            X[k] * np.cos(np.pi / N * (n + n0) * (2 * k + 1))
            for k in range(len(X))
        ])
    
    return x / (N / 2)  # Normalization factor

def mdct2d(block):
    """Apply MDCT to each row and then each column of the block."""
    return np.apply_along_axis(mdct, 0, np.apply_along_axis(mdct, 1, block))

def imdct2d(block):
    """Apply IMDCT to each column and then each row of the block."""
    return np.apply_along_axis(imdct, 0, np.apply_along_axis(imdct, 1, block))



def process_image(input_path, output_path, block_size=8):
    image = Image.open(input_path).convert('L')
    image_data = np.array(image)
    h, w = image_data.shape
    processed_data = np.zeros_like(image_data)

    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            block = image_data[i:i+block_size, j:j+block_size]

            # Apply 2D MDCT to the block
            mdct_block = mdct2d(block)

            # Apply 2D IMDCT to reconstruct the block
            reconstructed_block = imdct2d(mdct_block)

            # Store the reconstructed block in the processed data array
            processed_data[i:i+block_size, j:j+block_size] = np.round(reconstructed_block)

    # Clip values to be in the valid range [0, 255]
    processed_data = np.clip(processed_data, 0, 255).astype(np.uint8)

    # Create the output image
    output_image = Image.fromarray(processed_data)
    output_image.save(output_path)

# Example usage
if __name__ == "__main__":
    input_image_path = '/Users/mbhussain/Desktop/VC/input_images/'  # Replace with your input image path
    output_image_path = '/Users/mbhussain/Desktop/VC/output_images/'  # Output path for the processed image

    
for img_num in range(1,25):
    print(img_num)
    input_img = str(f'{input_image_path}{img_num}.jpeg')
    print(input_img)
    output_img = f'{output_image_path}{img_num}.jpeg'
    #quant_matrix = np.ones((4, 4)) * 10  # Example quantization matrix
    process_image(input_img, output_img)  # Set to False for DCT
#    process_image(input_image_path, output_image_path)
