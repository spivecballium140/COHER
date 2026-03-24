
import numpy as np

def file2stream(path):
    res = []
    with open(path, 'r') as f:
        for eve in f:
            res += [float(each) if '.' in each else int(each) for each in eve.strip('\n').split()]
    return res

def analyze(path, CE_Tnum, M_Jnum, M_OPTnum, Enum, Dnum, Cnum):
    print(f"Analyzing {path}...")
    fs = file2stream(path)
    
    # EtoD
    etod = []
    for _ in range(Enum * Dnum):
        etod.append(fs.pop(0))
    print(f"EtoD Distance: {min(etod)} - {max(etod)}")

    # DtoD
    dtod = []
    for _ in range(Dnum * Dnum):
        dtod.append(fs.pop(0))
    print(f"DtoD Distance: {min(dtod)} - {max(dtod)}")

    # MTask Time
    mtask_time = []
    for _ in range(M_Jnum * M_OPTnum):
        mtask_time.append(fs.pop(0))
    print(f"MTask Time: {min(mtask_time)} - {max(mtask_time)}")

    computations = []
    communications = []
    
    for i in range(CE_Tnum):
        computations.append(fs.pop(0))
        communications.append(fs.pop(0))
        
        # Precedence
        vec_num = fs.pop(0)
        for _ in range(vec_num): fs.pop(0)
        
        # Interact
        vec_num = fs.pop(0)
        for _ in range(vec_num): fs.pop(0)
        
        # Start_Pre
        vec_num = fs.pop(0)
        for _ in range(vec_num): fs.pop(0)
        
        # End_Pre
        vec_num = fs.pop(0)
        for _ in range(vec_num): fs.pop(0)
        
        fs.pop(0) # Job_Constraints

    print(f"Computation: {min(computations)} - {max(computations)}")
    print(f"Communication: {min(communications)} - {max(communications)}")

    # AvailDeviceList
    for i in range(M_Jnum * M_OPTnum):
        vec_num = fs.pop(0)
        for _ in range(vec_num): fs.pop(0)

    # AvailEdgeServerList
    for i in range(CE_Tnum):
        vec_num = fs.pop(0)
        for _ in range(vec_num): fs.pop(0)

    # EnergyList
    energy_list = []
    for _ in range(11):
        energy_list.append(fs.pop(0))
    print(f"Energy List: {min(energy_list)} - {max(energy_list)}")

    # Optional
    if len(fs) >= Dnum * 4:
        costs = [fs.pop(0) for _ in range(Dnum)]
        rels = [fs.pop(0) for _ in range(Dnum)]
        quals = [fs.pop(0) for _ in range(Dnum)]
        secs = [fs.pop(0) for _ in range(Dnum)]
        print(f"Device Cost: {min(costs)} - {max(costs)}")
        print(f"Device Reliability: {min(rels)} - {max(rels)}")
        print(f"Device Quality: {min(quals)} - {max(quals)}")
        print(f"Device Security: {min(secs)} - {max(secs)}")
    else:
        print("Optional attributes not found.")

if __name__ == "__main__":
    # Case 2 parameters from main.py: 200, 200, 5, 200, 300, 200
    analyze('./instances/data_matrix_200_seed1.txt', 200, 200, 5, 200, 300, 200)
