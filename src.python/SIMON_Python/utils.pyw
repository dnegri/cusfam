import glob
import os
import constants as cs

from model import *
import cecor_read


def read_cecor_file():
    cecor_read.read_cecor_file()

def check_extensions(path, extensions):
    extension_counter = 0
    for extension in extensions:
        if os.path.exists(path + extension):
            extension_counter += 1

    return len(extensions) <= extension_counter

def getPlantFiles(user):
    plants = []
    error_names = []
    file_names = glob.glob("{}*.XS".format(user.plant_directory))
    for file_name in file_names:
        basename = os.path.basename(file_name)
        extension_removed_path = os.path.splitext(file_name)[0]

        plant_found = False
        for plant in cs.DEFINED_PLANTS:
            if plant in basename:
                plants.append(os.path.splitext(basename)[0])
                plant_found = True
                break

        if not plant_found:

            error_names.append(file_name)
    return plants, error_names

#
# def getRestartFiles(user):
#     plants = []
#     error_names = []
#     file_names = glob.glob("{}*.RFA".format(user.restart_directory))
#     for file_name in file_names:
#         basename = os.path.basename(file_name)
#         extension_removed_path = os.path.splitext(file_name)[0]
#
#         if check_extensions(extension_removed_path, cs.RESTART_EXTENSIONS):
#             plant_found = False
#             for plant in cs.DEFINED_PLANTS:
#                 if plant in basename:
#                     plants.append(os.path.splitext(basename)[0])
#                     plant_found = True
#                     break
#
#             if not plant_found:
#                 error_names.append(file_name)
#         else:
#             error_names.append(file_name)
#     return plants, error_names
#

def getPDIL(power):

    R5_95_65 = [249.936, 5.2832]
    R5_65_50 = [91.44, 6.096]

    R4_95_65 = [381.0, 2.032]
    R4_65_50 = [320.04, 6.096]
    R4_50_42 = [228.6, 9.525]
    R4_42_20 = [152.4, 6.92727]

    R3_42_20 = [381.0, 9.0054]
    R3_20_0 = [182.88, 1.524]

    if 100 >= power > 95:
        pdil = [R5_95_65[0], R4_95_65[0], R3_42_20[0]]
    elif 95 >= power > 65:
        power_increment = 95-power
        pdil = [R5_95_65[0]-power_increment*R5_95_65[1],
                        R4_95_65[0]-power_increment*R4_95_65[1],
                        R3_42_20[0]]
    elif 65 >= power > 50:
        power_increment = 65-power
        pdil = [R5_65_50[0]-power_increment*R5_65_50[1],
                        R4_65_50[0]-power_increment*R4_65_50[1],
                        R3_42_20[0]]
    elif 50 >= power > 42:
        power_increment = 50-power
        pdil = [0,
                        R4_50_42[0]-power_increment*R4_50_42[1],
                        R3_42_20[0]]
    elif 42 >= power > 20:
        power_increment = 50-power
        pdil = [0,
                        R4_42_20[0]-power_increment*R4_42_20[1],
                        R3_42_20[0]-power_increment*R3_42_20[1]]
    elif 20 >= power >= 0:
        power_increment = 20 - power
        pdil = [0,
                        0,
                        R3_20_0[0] - power_increment * R3_20_0[1]]
    else:
        pdil = [381,
                        381,
                        381]
        #print("Power out of range {}".format(power))
    return pdil


def getPIDL_Rods(power, rod_pos, is_extended = True):

    pdils = getPDIL(power)
    if is_extended:
        diff_overlap = .10*381.0
    else:
        diff_overlap = .60*381.0

    r4_pdil = max(pdils[1], rod_pos[0]+diff_overlap)
    r3_pdil = max(pdils[2], rod_pos[1]+diff_overlap)

    return [pdils[0], r4_pdil, r3_pdil]

def getRestartFiles(user):

    plants = []
    error_names = []
    file_names = glob.glob("{}*.SMR".format(user.restart_directory))
    for file_name in file_names:
        basename = os.path.basename(file_name)
        extension_removed_path = os.path.splitext(file_name)[0]

        plant_found = False
        for plant in cs.DEFINED_PLANTS:
            if plant == basename[:len(plant)]:
                base_identifier = basename.split(".")[0]
                if base_identifier not in plants:
                    plants.append(base_identifier)
                plant_found = True
                break

        if not plant_found:
            error_names.append(file_name)

    return plants, error_names


def get_string_length(check_string):
    string_len = 0
    for i in range(len(check_string)):
        check_string_temp = check_string[:i+1]
        if check_string_temp.isalpha():
            string_len = i+1
    return string_len

def get_int_value(check_string):
    int_value = -1
    for i in range(len(check_string)):
        check_string_temp = check_string[:i]
        try:
            int(check_string_temp)
            int_value = int(check_string_temp)
        except ValueError:
            pass
    return int_value


def get_cecore_output(filename = ""):
    query = Cecor_Output.select().where(Cecor_Output.filename == filename).order_by(Cecor_Output.modified_date)
    return query

def get_all_inputs(class_find=None):

    if class_find:
        if class_find == ECP_Input:
            query = Calculations.select().where(Calculations.ecp_input.is_null(False)).order_by(-Calculations.modified_date)
        elif class_find == SD_Input:
            query = Calculations.select().where(Calculations.sd_input.is_null(False)).order_by(-Calculations.modified_date)
        elif class_find == RO_Input:
            query = Calculations.select().where(Calculations.ro_input.is_null(False)).order_by(-Calculations.modified_date)
        elif class_find == SDM_Input:
            query = Calculations.select().where(Calculations.sdm_input.is_null(False)).order_by(-Calculations.modified_date)
        elif class_find == CoastDownInput:
            query = Calculations.select().where(Calculations.coastdown_input.is_null(False)).order_by(-Calculations.modified_date)
        elif class_find == LifetimeInput:
            query = Calculations.select().where(Calculations.lifetime_input.is_null(False)).order_by(-Calculations.modified_date)
    else:
        query = Calculations.select().order_by(-Calculations.modified_date)

    return query

def get_default_input(user, class_find):

    try:

        query = Calculations.select().where(Calculations.filename == cs.RECENT_CALCULATION).order_by(-Calculations.modified_date)
        calculation_final = None
        input_final = None
        for calculation_object in query:
            if class_find == ECP_Input:
                if calculation_object.ecp_input:
                    calculation_final = calculation_object
                    input_final = calculation_object.ecp_input
                    break
            elif class_find == SD_Input:
                if calculation_object.sd_input:
                    calculation_final = calculation_object
                    input_final = calculation_object.sd_input
                    break
            elif class_find == RO_Input:
                if calculation_object.ro_input:
                    calculation_final = calculation_object
                    input_final = calculation_object.ro_input
                    break
            elif class_find == SDM_Input:
                if calculation_object.sdm_input:
                    calculation_final = calculation_object
                    input_final = calculation_object.sdm_input
                    break
            elif class_find == CoastDownInput:
                if calculation_object.coastdown_input:
                    calculation_final = calculation_object
                    input_final = calculation_object.coastdown_input
                    break
            elif class_find == LifetimeInput:
                if calculation_object.lifetime_input:
                    calculation_final = calculation_object
                    input_final = calculation_object.lifetime_input
                    break
    except:
        return None, None

    return calculation_final, input_final

def get_last_input(user, class_find):

    query = Calculations.select().where(Calculations.user == user).order_by(-Calculations.modified_date)
    calculation_final = None
    input_final = None
    for calculation_object in query:
        if class_find == ECP_Input:
            if calculation_object.ecp_input:
                calculation_final = calculation_object
                input_final = calculation_object.ecp_input
                break
        elif class_find == SD_Input:
            if calculation_object.sd_input:
                calculation_final = calculation_object
                input_final = calculation_object.sd_input
                break
        elif class_find == RO_Input:
            if calculation_object.ro_input:
                calculation_final = calculation_object
                input_final = calculation_object.ro_input
                break
        elif class_find == SDM_Input:
            if calculation_object.sdm_input:
                calculation_final = calculation_object
                input_final = calculation_object.sdm_input
                break
        elif class_find == CoastDownInput:
            if calculation_object.coastdown_input:
                calculation_final = calculation_object
                input_final = calculation_object.coastdown_input
                break
        elif class_find == LifetimeInput:
            if calculation_object.lifetime_input:
                calculation_final = calculation_object
                input_final = calculation_object.lifetime_input
                break

    return calculation_final, input_final



def delete_calculations(calculation_object):
    if calculation_object.ecp_input:
        calculation_object.ecp_input.delete_instance()

    if calculation_object.sd_input:
        calculation_object.sd_input.delete_instance()

    if calculation_object.ro_input:
        calculation_object.ro_input.delete_instance()

    if calculation_object.sdm_input:
        calculation_object.sdm_input.delete_instance()

    if calculation_object.ecp_output:
        calculation_object.ecp_output.delete_instance()

    if calculation_object.sd_output:
        calculation_object.sd_output.delete_instance()

    if calculation_object.ro_output:
        calculation_object.ro_output.delete_instance()

    if calculation_object.sdm_output:
        calculation_object.sdm_output.delete_instance()

    calculation_object.delete_instance()