import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == '__main__':
    # Set the random seed for reproducibility
    np.random.seed(42)

    # Generate a 224x224 image with values greater than 1 and less than 0
    image = np.random.randn(224, 224, 3)

    # Modify the first and second elements
    image[0, 0, 0] = 2
    image[0, 0, 1] = -1

    # Print the modified values
    print(f'Original first element: {image[0, 0, 0]}')
    print(f'Original second element: {image[0, 0, 1]}')

    # Display the image using imshow
    plt.imshow(image)  # Normalize to [0, 1] for display
    plt.axis('off')  # Hide axes
    plt.gca().set_position([0, 0, 1, 1])  # Remove padding
    plt.gcf().set_size_inches(224 / plt.gcf().dpi, 224 / plt.gcf().dpi)  # Set figure size to match image size
    plt.savefig('image.png', bbox_inches='tight', pad_inches=0)  # Save the displayed image

    # Load the saved image
    # Open the image
    image_pil = Image.open('image.png').convert('RGB')  # Convert to RGB to ensure no alpha channel

    # Resize the image to 224x224
    image_resized = image_pil.resize((224, 224))

    # Convert the image to a NumPy array
    image_array = np.array(image_resized)

    # Convert the array to float32
    image_float32 = image_array.astype(np.float32)

    # Normalize the array values to the range [0, 1]
    loaded_image = image_float32 / 255.0

    # Restore the loaded image values to the original range
    loaded_image = loaded_image * (image.max() - image.min()) + image.min()

    # Print the loaded values
    print(f'Loaded first element: {loaded_image[0, 0, 0]}')
    print(f'Loaded second element: {loaded_image[0, 0, 1]}')

    # Calculate the difference
    difference = np.abs(image - loaded_image)
    max_difference = np.max(difference)

    print(f'Max difference: {max_difference}')