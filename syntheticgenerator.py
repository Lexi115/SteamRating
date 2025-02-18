import pandas as pd
import random

# Define the exact values for hardware specifications
cpu_cores = {'Basic Use': (2, 4), 'Office': (4, 6), 'Gaming': (6, 8, 12), 'Editing': (8, 12, 16)}
gpu_vram = {'Basic Use': (1, 2), 'Office': (2, 4), 'Gaming': (6, 8, 12, 16), 'Editing': (16, 24, 32)}
ram_gb = {'Basic Use': (4, 8), 'Office': (8, 16), 'Gaming': (16, 32), 'Editing': (32, 64)}
ssd_gb = {'Basic Use': (128, 256), 'Office': (256, 512), 'Gaming': (512, 1024, 2048), 'Editing': (2048, 4096, 8192)}
psu_watt = {'Basic Use': (200, 300), 'Office': (300, 450), 'Gaming': (450, 600, 750, 1000), 'Editing': (750, 1000, 2000)}

# Function to generate a dataset
def generate_dataset(num_samples=30000):
    data = []
    for _ in range(num_samples):
        usage = random.choice(['Basic Use', 'Office', 'Gaming', 'Editing'])
        cpu = random.choice(cpu_cores[usage])
        gpu = random.choice(gpu_vram[usage])
        ram = random.choice(ram_gb[usage])
        ssd = random.choice(ssd_gb[usage])
        psu = random.choice(psu_watt[usage])
        data.append([cpu, gpu, ram, ssd, psu, usage])

    columns = ['CPU Cores', 'GPU VRAM (GB)', 'RAM (GB)', 'SSD (GB)', 'PSU Wattage', 'Usage Type']
    return pd.DataFrame(data, columns=columns)

# Generate the dataset with 30,000 samples
df = generate_dataset(30000)

# Save to CSV
df.to_csv('resources\\synthetic_pc_classification_dataset_large.csv', index=False)
