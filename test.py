import numpy as np
import matplotlib.pyplot as plt
image_path = r"C:\Users\User\Documents\Pixel_model_10_reflectors\output_avi\results\0\farfield_2400.npy"

image = np.load(image_path)
# image.reshape([34,34])
plt.imshow(10*np.log10(image[:,:,0]), cmap='jet')
plt.clim(-10,10)
plt.show()
