import os

def file2stream(path):
    res = []
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        for eve in f:
            # 兼容 float 和 int 的读取
            res += [float(each) if '.' in each else int(each) for each in eve.strip('\n').split()]
    return res

def verify_file(filepath, config):
    CE_Tnum, M_Jnum, M_OPTnum, Enum, Dnum, Cnum = config
    print(f"Checking {filepath} with config: {config}")
    
    fs = file2stream(filepath)
    if fs is None:
        print(f"Error: File {filepath} not found.")
        return False

    total_len = len(fs)
    print(f"Total tokens in file: {total_len}")
    
    try:
        # 1. EtoD_Distance
        consumed = 0
        step_len = Enum * Dnum
        for _ in range(step_len): fs.pop(0)
        consumed += step_len
        # print(f"EtoD_Distance: {step_len} items")

        # 2. DtoD_Distance
        step_len = Dnum * Dnum
        for _ in range(step_len): fs.pop(0)
        consumed += step_len
        # print(f"DtoD_Distance: {step_len} items")

        # 3. MTask_Time
        step_len = M_Jnum * M_OPTnum
        for _ in range(step_len): fs.pop(0)
        consumed += step_len
        # print(f"MTask_Time: {step_len} items")

        # 4. CETask_Property
        ce_task_items = 0
        for _ in range(CE_Tnum):
            fs.pop(0); fs.pop(0) # Computation, Communication
            ce_task_items += 2
            
            # 4 vectors: Precedence, Interact, Start_Pre, End_Pre
            for _ in range(4):
                vec_num = int(fs.pop(0))
                ce_task_items += 1
                for _ in range(vec_num): 
                    fs.pop(0)
                    ce_task_items += 1
            
            fs.pop(0) # Job_Constraints
            ce_task_items += 1
        consumed += ce_task_items
        # print(f"CETask_Property: {ce_task_items} items")

        # 5. AvailDeviceList
        avail_dev_items = 0
        for _ in range(M_Jnum * M_OPTnum):
            vec_num = int(fs.pop(0))
            avail_dev_items += 1
            for _ in range(vec_num): 
                fs.pop(0)
                avail_dev_items += 1
        consumed += avail_dev_items
        # print(f"AvailDeviceList: {avail_dev_items} items")

        # 6. AvailEdgeServerList
        avail_edge_items = 0
        for _ in range(CE_Tnum):
            vec_num = int(fs.pop(0))
            avail_edge_items += 1
            for _ in range(vec_num): 
                fs.pop(0)
                avail_edge_items += 1
        consumed += avail_edge_items
        # print(f"AvailEdgeServerList: {avail_edge_items} items")

        # 7. EnergyList
        step_len = 11
        for _ in range(step_len): fs.pop(0)
        consumed += step_len
        # print(f"EnergyList: {step_len} items")

        # 8. New Attributes (Optional but expected for new files)
        # Device Cost, Reliability, Quality, Security
        # Each is Dnum items
        remaining = len(fs)
        expected_new_attrs = Dnum * 4
        
        if remaining == expected_new_attrs:
            print(f"Found exactly {remaining} remaining items, matching 4 new attributes.")
            for _ in range(expected_new_attrs): fs.pop(0)
            consumed += expected_new_attrs
        elif remaining == 0:
             print("No new attributes found (older format).")
        else:
            print(f"Warning: Remaining items {remaining} does not match expected {expected_new_attrs} for new attributes.")
            
        if len(fs) == 0:
            print(f"SUCCESS: File {filepath} parsed completely and correctly.")
            return True
        else:
            print(f"FAILURE: File {filepath} has {len(fs)} unconsumed items.")
            return False

    except IndexError:
        print(f"FAILURE: Unexpected end of file while parsing {filepath}. Consumed {consumed} items.")
        return False
    except Exception as e:
        print(f"FAILURE: An error occurred parsing {filepath}: {e}")
        return False

def main():
    # CE_Tnum, M_Jnum, M_OPTnum, Enum, Dnum, Cnum
    configs = {
        '600': (600, 600, 5, 600, 600, 600),
        '800': (800, 800, 5, 800, 800, 800),
        '1000': (1000, 1000, 5, 1000, 1000, 1000)
    }

    files_to_check = [
        ('instances/data_matrix_600_seed1.txt', '600'),
        ('instances/data_matrix_800_seed1.txt', '800'),
        ('instances/data_matrix_1000_seed1.txt', '1000')
    ]

    all_passed = True
    for filepath, scale in files_to_check:
        print("-" * 50)
        if not verify_file(filepath, configs[scale]):
            all_passed = False
    
    print("-" * 50)
    if all_passed:
        print("All files passed verification.")
    else:
        print("Some files failed verification.")

if __name__ == "__main__":
    main()

