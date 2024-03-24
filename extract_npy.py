import numpy as np

# Load the .npy file containing a dictionary
loaded_data = np.load(r'C:\Users\LAPTOP88\Downloads\test_HairCLIPv2-alt\latents_3.npy', allow_pickle=True).item()

# Access and print the contents of the dictionary
for key, value in loaded_data.items():
    target_dir='test_sg3_output/W3/'+key.split('.')[0]+'.npy'
    np.save(target_dir, value[0])