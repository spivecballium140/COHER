import os
import random
import glob
#生成数据文件
def file2stream(path):
    res = []
    with open(path, 'r') as f:
        for eve in f:
            res += [float(each) if '.' in each else int(each) for each in eve.strip('\n').split()]
    return res

def check_and_update_file(filepath, config):
    CE_Tnum, M_Jnum, M_OPTnum, Enum, Dnum, Cnum = config
    
    # Read file content
    fs = file2stream(filepath)
    original_len = len(fs)
    
    # Simulate consumption to find where the "standard" data ends
    try:
        # EtoD
        for _ in range(Enum * Dnum): fs.pop(0)
        # DtoD
        for _ in range(Dnum * Dnum): fs.pop(0)
        # MTask_Time
        for _ in range(M_Jnum * M_OPTnum): fs.pop(0)
        
        # CE Tasks
        for _ in range(CE_Tnum):
            fs.pop(0); fs.pop(0) # Comp, Comm
            # 4 vectors
            for _ in range(4):
                vec_num = int(fs.pop(0))
                for _ in range(vec_num): fs.pop(0)
            fs.pop(0) # Job Constraints
            
        # AvailDeviceList
        for _ in range(M_Jnum * M_OPTnum):
            vec_num = int(fs.pop(0))
            for _ in range(vec_num): fs.pop(0)
            
        # AvailEdgeServerList
        for _ in range(CE_Tnum):
            vec_num = int(fs.pop(0))
            for _ in range(vec_num): fs.pop(0)
            
        # EnergyList
        for _ in range(11): fs.pop(0)
        
        # Check if remaining data exists
        if len(fs) > 0:
            print(f"File {filepath} already appears to have extra data ({len(fs)} items). Skipping.")
            return
            
    except IndexError:
        print(f"File {filepath} is corrupted or shorter than expected. Skipping.")
        return

    print(f"Updating {filepath} with new attributes...")
    
    # Generate new data
    new_data = []
    
    # Device Cost: uniform(300, 3000)
    new_data.extend([round(random.uniform(300, 3000), 2) for _ in range(Dnum)])
    
    # Device Reliability: uniform(0.85, 0.99)
    new_data.extend([round(random.uniform(0.85, 0.99), 4) for _ in range(Dnum)])
    
    # Device Quality: uniform(0.85, 0.99)
    new_data.extend([round(random.uniform(0.85, 0.99), 4) for _ in range(Dnum)])
    
    # Device Security: uniform(0.85, 0.99)
    new_data.extend([round(random.uniform(0.85, 0.99), 4) for _ in range(Dnum)])
    
    # Append to file
    with open(filepath, 'a') as f:
        f.write('\n\n') # Add some separation
        # Write in chunks for readability
        chunk_size = 10
        for i in range(0, len(new_data), chunk_size):
            chunk = new_data[i:i+chunk_size]
            line = '\t'.join(map(str, chunk))
            f.write(line + '\n')
            
    print(f"Successfully updated {filepath}.")

def generate_full_file(filepath, config):
    CE_Tnum, M_Jnum, M_OPTnum, Enum, Dnum, Cnum = config
    print(f"Generating new instance file: {filepath} with config {config}")
    
    data = []
    
    # 1. EtoD_Distance (Enum * Dnum)
    data.extend([round(random.uniform(100, 5000), 3) for _ in range(Enum * Dnum)])
    
    # 2. DtoD_Distance (Dnum * Dnum)
    data.extend([round(random.uniform(100, 5000), 3) for _ in range(Dnum * Dnum)])
    
    # 3. MTask_Time (M_Jnum * M_OPTnum)
    data.extend([round(random.uniform(10, 500), 3) for _ in range(M_Jnum * M_OPTnum)])
    
    # 4. CETask_Property (CE_Tnum)
    for _ in range(CE_Tnum):
        # Computation, Communication
        data.append(round(random.uniform(100, 2000), 3))
        data.append(round(random.uniform(10, 500), 3))
        
        # 4 vectors: Precedence, Interact, Start_Pre, End_Pre
        for _ in range(4):
            # Generate sparse connections
            if random.random() < 0.2:
                vec_num = random.randint(1, 3)
                data.append(vec_num)
                # Random task indices
                data.extend(random.sample(range(CE_Tnum), vec_num))
            else:
                data.append(0)
        
        # Job_Constraints (0 or 1)
        data.append(random.randint(0, 1))

    # 5. AvailDeviceList (M_Jnum * M_OPTnum)
    for _ in range(M_Jnum * M_OPTnum):
        # Assign random devices (subset of Dnum)
        num_dev = random.randint(1, min(10, Dnum))
        data.append(num_dev)
        data.extend(random.sample(range(Dnum), num_dev))
        
    # 6. AvailEdgeServerList (CE_Tnum)
    for _ in range(CE_Tnum):
        # Assign random edge servers (subset of Enum)
        num_edge = random.randint(1, min(5, Enum))
        data.append(num_edge)
        data.extend(random.sample(range(Enum), num_edge))
        
    # 7. EnergyList (11)
    data.extend([round(random.uniform(10, 100), 3) for _ in range(11)])
    
    # 8-11. New Attributes (Cost, Reliability, Quality, Security)
    # Device Cost
    data.extend([round(random.uniform(300, 3000), 2) for _ in range(Dnum)])
    # Device Reliability
    data.extend([round(random.uniform(0.85, 0.99), 4) for _ in range(Dnum)])
    # Device Quality
    data.extend([round(random.uniform(0.85, 0.99), 4) for _ in range(Dnum)])
    # Device Security
    data.extend([round(random.uniform(0.85, 0.99), 4) for _ in range(Dnum)])
    
    # Write to file
    with open(filepath, 'w') as f:
        # Write in chunks
        chunk_size = 20
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i+chunk_size]
            line = '\t'.join(map(str, chunk))
            f.write(line + '\n')
            
    print(f"Generated {filepath} with {len(data)} values.")

def main():
    # Config map based on main.py logic
    # (CE_Tnum, M_Jnum, M_OPTnum, Enum, Dnum, Cnum)
    configs = {
        '200': (200, 200, 5, 200, 300, 200),
        '300': (300, 300, 5, 300, 300, 300),
        '400': (400, 400, 5, 400, 300, 400),
        '500': (500, 500, 5, 500, 500, 500),
        '600': (600, 600, 5, 600, 600, 600),
        '800': (800, 800, 5, 800, 800, 800),
        '1000': (1000, 1000, 5, 1000, 1000, 1000)
    }
    
    # Generate/Update specific scales
    target_scales = ['600', '800', '1000']
    
    # First, handle generation of new files
    for scale in target_scales:
        if scale in configs:
            for seed in range(1, 4):
                filepath = f'instances/data_matrix_{scale}_seed{seed}.txt'
                if not os.path.exists(filepath):
                    generate_full_file(filepath, configs[scale])
                else:
                    print(f"File {filepath} already exists. Skipping generation.")
        else:
            print(f"Config for scale {scale} not found.")
    
    # Then run update on all matching files (including existing ones)
    files = glob.glob('instances/data_matrix_*_seed*.txt')
    
    for filepath in files:
        filename = os.path.basename(filepath)
        # Extract scale (200, 300, etc)
        parts = filename.split('_')
        if len(parts) >= 3:
            scale = parts[2] # data, matrix, 200, seed1.txt
            if scale in configs:
                check_and_update_file(filepath, configs[scale])
            else:
                print(f"Skipping {filename}: Unknown scale {scale}")

if __name__ == "__main__":
    main()

