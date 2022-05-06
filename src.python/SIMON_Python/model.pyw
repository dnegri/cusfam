from peewee import *
import constants as cs
from datetime import date
import Definitions as df

db = SqliteDatabase('simon.db')

class BaseModel(Model):
    class Meta:
        database = db


class User(BaseModel):
    username = CharField(unique=True)
    last_login = DateTimeField(null=True)

    modified = BooleanField()

    working_directory = CharField(null=True)
    plant_directory = CharField(null=True)
    plant_file = CharField(null=True)
    restart_directory = CharField(null=True)
    restart_file = CharField(null=True)
    cycle_number = CharField(null=True)
    cecor_directory = CharField(null=True)


class LoginUser(BaseModel):
    username = CharField(unique=True)
    login_user = ForeignKeyField(User, backref='login')


class InputModel(BaseModel):
    calculation_type = CharField(default="User Input")
    snapshot_table = TextField(default="")
    snapshot_text = CharField(default="Load Data")


class MonitoringInput(InputModel):
    rst_step = IntegerField
    core_burnup = FloatField()
    asi = FloatField()
    fxy = FloatField()
    rel_power = FloatField()


class ECP_Input(InputModel):

    #Required Object
    search_type = IntegerField(default=0)

    #NDR
    #bs_ndr_date_time = DateTimeField()
    bs_ndr_date = DateField()
    bs_ndr_time = TimeField()
    bs_ndr_power = FloatField(default=100.0)
    bs_ndr_burnup = FloatField(default=0.0)
    bs_ndr_average_temperature = FloatField(default=272.12)
    bs_ndr_target_eigen = FloatField(default=1.0)
    bs_ndr_bank_position_P = FloatField(default=190.5)
    bs_ndr_bank_position_5 = FloatField(default=190.5)
    bs_ndr_bank_position_4 = FloatField(default=381.0)

    # Rod Search
    #as_ndr_date_time = DateTimeField()
    as_ndr_delta_time = FloatField(default=100.0)
    as_ndr_boron_concentration = FloatField(default=1000.0)

    as_ndr_bank_position_P = FloatField(default=190.5)
    as_ndr_bank_position_5 = FloatField(default=190.5)
    as_ndr_bank_position_4 = FloatField(default=381.0)

class SD_Input(InputModel):

    ndr_burnup = FloatField(default=0.0)
    ndr_target_keff = FloatField(default=1.0)
    ndr_power_ratio = FloatField(default=3.0)
    ndr_power_asi = FloatField(default=0.010)

class RPCS_Input(InputModel):

    calculation_type = IntegerField()

    search_type = IntegerField()

    # Snapshot
    ss_snapshot_file = CharField()

    # ndr
    ndr_burnup = FloatField()
    ndr_target_keff = FloatField()
    ndr_power = FloatField()

class RO_Input(InputModel):

    #ndr_cal_type = CharField()

    ndr_burnup = FloatField(default=12000.0)
    ndr_power = FloatField(default=1.0)

    ndr_time = FloatField(default=30.0)

    ndr_bank_position_5 = FloatField(default=190.5)
    ndr_bank_position_4 = FloatField(default=381.0)
    ndr_bank_position_3 = FloatField(default=381.0)
    ndr_bank_position_P = FloatField(default=190.5)

    ndr_target_keff = FloatField(default=1.0)
    ndr_power_ratio = FloatField(default=3.0)
    ndr_asi = FloatField(default=0.010)
    ndr_end_power = FloatField(default=100.0)

class SDM_Input(InputModel):
    #ndr
    ndr_mode_selection = CharField(default="Mode 1, 2")
    ndr_burnup = FloatField(default=0.0)
    ndr_power = FloatField(default=100.0)

    ndr_stuckrod1_x = IntegerField(default=-1)
    ndr_stuckrod1_y = IntegerField(default=-1)
    ndr_stuckrod2_x = IntegerField(default=-1)
    ndr_stuckrod2_y = IntegerField(default=-1)


class CoastDownInput(InputModel):

    #Calculation Options
    calculation_type = IntegerField()

    #Calculation Object
    search_type = IntegerField()

    #Snapshot
    ss_snapshot_file = CharField()

    #ndr
    ndr_burnup = FloatField()
    ndr_target_asi = FloatField()
    ndr_stopping_criterion = FloatField()
    ndr_depletion_interval = FloatField()

    ndr_bank_position_5 = FloatField()
    ndr_bank_position_4 = FloatField()
    ndr_bank_position_3 = FloatField()
    ndr_bank_position_P = FloatField()

class LifetimeInput(InputModel):

    #Calculation Options
    calculation_type = IntegerField()

    #Snapshot
    ss_snapshot_file = CharField()

    #ndr
    ndr_burnup = FloatField()
    ndr_power = FloatField()
    ndr_stopping_criterion = FloatField()
    ndr_depletion_interval = FloatField()

    ndr_bank_position_5 = FloatField()
    ndr_bank_position_4 = FloatField()
    ndr_bank_position_3 = FloatField()
    ndr_bank_position_P = FloatField()

class OutputModel(BaseModel):
    pass

class ECP_Output(OutputModel):
    success = BooleanField(default=False)
    table = TextField(default="")

class SD_Output(OutputModel):
    success = BooleanField(default=False)
    table = TextField(default="")

class RO_Output(OutputModel):
    success = BooleanField(default=False)
    table = TextField(default="")

class SDM_Output(OutputModel):

    m1_success = BooleanField(default=False)
    m1_core_burnup = FloatField(default=0.0)
    m1_temperature = FloatField(default=df.inlet_temperature)
    m1_cea_configuration = CharField(default="")
    m1_stuck_rod = CharField(default="")
    m1_n1_worth = FloatField(default=0.0)
    m1_defect_worth = FloatField(default=0.0)
    m1_required_worth = FloatField(default=0.0)
    m1_sdm_worth = FloatField(default=0.0)

    m2_success = BooleanField(default=False)
    m2_core_burnup = FloatField(default=0.0)
    m2_temperature = FloatField(default=0.0)
    m2_cea_configuration = CharField(default="")
    m2_stuck_rod = CharField(default="")
    m2_required_worth = FloatField(default=0.0)
    m2_required_cbc = FloatField(default=0.0)


class Cecor_Output(BaseModel):
    filename = CharField(default="")
    table = TextField(default="")
    modified_date = DateTimeField()

class Calculations(BaseModel):
    user = ForeignKeyField(User, backref='user')

    filename = CharField(default=cs.RECENT_CALCULATION)
    comments = CharField(default="")

    calculation_type = CharField()
    created_date = DateTimeField()
    modified_date = DateTimeField()

    saved = BooleanField(default=False)

    ecp_input = ForeignKeyField(ECP_Input, backref='ecp_input', null=True)
    sd_input = ForeignKeyField(SD_Input, backref='sd_input', null=True)
    ro_input = ForeignKeyField(RO_Input, backref='ro_input', null=True)
    sdm_input = ForeignKeyField(SDM_Input, backref='sdm_input', null=True)

    ecp_output = ForeignKeyField(ECP_Output, backref='ecp_output', null=True)
    sd_output = ForeignKeyField(SD_Output, backref='sd_output', null=True)
    ro_output = ForeignKeyField(RO_Output, backref='ro_output', null=True)
    sdm_output = ForeignKeyField(SDM_Output, backref='sdm_output', null=True)
