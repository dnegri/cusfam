from peewee import *
import constants as cs
from datetime import date

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
    snapshot_directory = CharField(null=True)


class LoginUser(BaseModel):
    username = CharField(unique=True)
    login_user = ForeignKeyField(User, backref='login')


class InputModel(BaseModel):
    pass


class MonitoringInput(InputModel):
    rst_step = IntegerField
    core_burnup = FloatField()
    asi = FloatField()
    fxy = FloatField()
    rel_power = FloatField()


class ECP_Input(InputModel):

    #Required Object
    search_type = IntegerField()

    #NDR
    #bs_ndr_date_time = DateTimeField()
    bs_ndr_date = DateField()
    bs_ndr_time = TimeField()
    bs_ndr_power = FloatField()
    bs_ndr_burnup = FloatField()
    bs_ndr_average_temperature = FloatField()
    bs_ndr_target_eigen = FloatField()
    bs_ndr_bank_position_P = FloatField()
    bs_ndr_bank_position_5 = FloatField()
    bs_ndr_bank_position_4 = FloatField()

    # Rod Search
    #as_ndr_date_time = DateTimeField()
    as_ndr_delta_time = FloatField()
    as_ndr_boron_concentration = FloatField()

    as_ndr_bank_position_P = FloatField()
    as_ndr_bank_position_5 = FloatField()
    as_ndr_bank_position_4 = FloatField()

class SD_Input(InputModel):

    ndr_burnup = FloatField()
    ndr_target_keff = FloatField()
    ndr_power_ratio = FloatField()
    ndr_power_asi = FloatField()

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

    ndr_cal_type = CharField()

    ndr_burnup = FloatField()
    ndr_power = FloatField()

    ndr_time = FloatField()

    ndr_bank_position_5 = FloatField()
    ndr_bank_position_4 = FloatField()
    ndr_bank_position_3 = FloatField()
    ndr_bank_position_P = FloatField()


    ndr_target_keff = FloatField()
    ndr_power_ratio = FloatField()
    ndr_asi = FloatField()
    ndr_end_power = FloatField()

class SDM_Input(InputModel):
    #ndr
    ndr_burnup = FloatField()
    ndr_mode_selection = CharField()

    ndr_stuckrod1_x = IntegerField()
    ndr_stuckrod1_y = IntegerField()
    ndr_stuckrod2_x = IntegerField()
    ndr_stuckrod2_y = IntegerField()


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
    table_values = TextField()

class SD_Output(OutputModel):
    table_values = TextField()
    p1d_values = TextField()
    rod_values = TextField()
    p2d_values = TextField()

class RO_Output(OutputModel):
    table_values = TextField()
    p1d_values = TextField()
    rod_values = TextField()
    p2d_values = TextField()


class SDM_Output(OutputModel):
    m1_success = BooleanField()
    m1_core_burnup = FloatField()
    m1_temperature = FloatField()
    m1_cea_configuration = CharField()
    m1_stuck_rod = CharField()
    m1_n1_worth = FloatField()
    m1_defect_worth = FloatField()
    m1_required_worth = FloatField()
    m1_sdm_worth = FloatField()

    m2_success = BooleanField()
    m2_core_burnup = FloatField()
    m2_temperature = FloatField()
    m2_cea_configuration = CharField()
    m2_stuck_rod = CharField()
    m2_required_worth = FloatField()
    m2_required_cbc = FloatField()


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



"""
db.connect()
db.create_tables([Person, Pet])
uncle_bob = Person.create(name="Bob", birthday=date(1960, 1, 15))
grandma = Person.create(name="Grandma", birthday=date(1960, 1, 15))

grandma.name = 'Grandma L.'
grandma.save()

bob_kitty = Pet.create(owner=uncle_bob, name="Kitty", animal_type='cat')
grandma_dog = Pet.create(owner=grandma, name="Dog", animal_type='dog')

grandma_dog.delete_instance()

grandma_found = Person.select().where(Person.name == 'Grandma L.').get()

print(grandma_found.name)
for person in Person.select():
    print(person.name, person.pets.count(), 'pets')

query = (Person
         .select(Person, fn.Count(Pet.id).alias('pet_count'))
         .join(Pet, JOIN.LEFT_OUTER)
         .group_by(Person)
         .order_by(Person.name))

for person in query:
    print(person.name, person.pets.count(), 'pets')
"""