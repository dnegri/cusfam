
block_job_name = "JOB NAME:"
block_snapshot_name = "1CECOR "
key_snapshot_name = "SNAPSHOT"
key_brunup = "EFPH"
block_rod1 = "ROD BANK NUMBER  1"
block_rod2 = "ROD BANK NUMBER  2"
block_rod3 = "ROD BANK NUMBER  3"
block_rod4 = "ROD BANK NUMBER  4"
block_rod5 = "ROD BANK NUMBER  5"
block_rod6 = "ROD BANK NUMBER  6"
block_rod7 = "ROD BANK NUMBER  7"
block_rod8 = "ROD BANK NUMBER  8"

key_rod3 = "R3"
key_rod4 = "R4"
key_rod5 = "R5"
key_rodP = "P"

block_core_power = "CORE POWER AS CALCULATED BY CECOR"
block_hfp = "NAME PLATE POWER AT 100 PCT. POWER"

store_job_name = "JOB_NAME"
store_current_snapshot = "SNAPSHOT"
store_current_burnup = "BURNUP"
store_current_hfp = "HFP"
store_current_power = "CORE_POWER"
store_snapshot_identifier = "SNAPSHOT_ID_"

def get_job_name(lines, l_i, storage):
    "return first element in next line"
    line = lines[l_i + 1]
    split_line=line.split()
    # actions.pop(block_job_name)
    storage[store_job_name] = split_line[0]
    return [split_line[0], ]

def get_current_job_burnup(lines, l_i, storage):
    "return element after snapeshot"
    line = lines[l_i]
    split_line=line.split()
    r_strings = []
    for w_i, word in enumerate(split_line):
        if key_snapshot_name in word:
            r_strings.append(split_line[w_i+1])
        if key_brunup in word:
            try:
                int(split_line[w_i+1][1:])
                r_strings.append(split_line[w_i+1][1:])
            except:
                int(split_line[w_i+2])
                r_strings.append(split_line[w_i+2])

    storage[store_current_snapshot] = r_strings[0]
    storage[store_snapshot_identifier+r_strings[0]] = r_strings[0]
    storage[storage[store_current_snapshot]+","+store_current_burnup] = r_strings[1]

    return r_strings

def get_current_rod(lines, l_i, storage):
    "return element after snapeshot"
    line = lines[l_i]
    split_line = line.split()
    r_strings = [split_line[9],]
    if block_rod3 in line:
        storage[storage[store_current_snapshot]+","+key_rod3] = split_line[9]
        storage[key_rod3] = split_line[9]
    if block_rod4 in line:
        storage[storage[store_current_snapshot]+","+key_rod4] = split_line[9]
        storage[key_rod4] = split_line[9]
    if block_rod5 in line:
        storage[storage[store_current_snapshot]+","+key_rod5] = split_line[9]
        storage[key_rod5] = split_line[9]
    if block_rod6 in line:
        storage[storage[store_current_snapshot]+","+key_rodP] = split_line[9]
        storage[key_rodP] = split_line[9]

    return r_strings

def get_current_core_power(lines, l_i, storage):

    line2 = lines[l_i]
    split_line2 = line2.split()

    # print("core power", float(split_line2[-2])/storage[storage[store_current_snapshot]+","+store_current_hfp])

    storage[storage[store_current_snapshot]+","+store_current_power] = float(split_line2[-2])/storage[storage[store_current_snapshot]+","+store_current_hfp]


def get_current_hfp(lines, l_i, storage):

    line2 = lines[l_i]
    split_line2 = line2.split()

    # print("hfp", float(split_line2[-2]))
    storage[storage[store_current_snapshot]+","+store_current_hfp] = float(split_line2[-2])


cecor_read_actions = {
    block_job_name:get_job_name,
    block_snapshot_name: get_current_job_burnup,
    block_rod3: get_current_rod,
    block_rod4: get_current_rod,
    block_rod5: get_current_rod,
    block_rod6: get_current_rod,
    block_core_power: get_current_core_power,
    block_hfp: get_current_hfp,
}

data_storage = {}

def read_cecor_file(file_path="C:/simon/snapshot/yg3_cecore.out"):
    file1 = open(file_path, 'r')
    lines = file1.readlines()
    storage = {}
    for l_i, line in enumerate(lines):

        for k_i, key in enumerate(cecor_read_actions):
            if key in line:
                print(key, cecor_read_actions[key](lines, l_i, storage) )

    snapshot_ids = []
    for key in storage:
        if store_snapshot_identifier in key:
            snapshot_ids.append(storage[key])
            print(storage[key])
    # len(store_snapshot_identifier):

    print("Job Name: ", storage[store_job_name])
    final_csv = []
    print("Power, R3, R4, R5, RP")
    for snapshot_id in snapshot_ids:

        pd = storage[snapshot_id+","+store_current_power]
        r3v = storage[snapshot_id+","+key_rod3]
        r4v = storage[snapshot_id+","+key_rod4]
        r5v = storage[snapshot_id+","+key_rod5]
        rPv = storage[snapshot_id+","+key_rodP]

        final_csv.append([pd, float(r3v), float(r4v), float(r5v), float(rPv)])
        print("{:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(pd, float(r3v), float(r4v), float(r5v), float(rPv)))

if __name__ == "__main__":
    read_cecor_file()