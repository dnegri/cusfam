
_BLANK_ASM_ = " "

_OPR1000_XPOS_ = 15
_OPR1000_YPOS_ = 15
_APR1400_XPOS_ = 17
_ARP1400_YPOS_ = 17

_POSITION_NONE_ =-1
_POSITION_CORE_ = 0
_POSITION_SFP01_= 1
_POSITION_SFP02_= 2
_POSITION_SFP_  = 3
_POSITION_NAME_ = [ "CORE", "SFP01", "SFP02"]

rgbDummy = ["CC","CC","CC"]
rgbSet = [ ["FF","FF","CC"] ,
           ["CC","FF","FF"] ,
           ["FF","CC","FF"] ,
           ["FF","CC","99"] ,
           ["99","FF","CC"] ,
           ["CC","99","FF"] ,
           ["FF","CC","CC"] ,
           ["CC","FF","CC"] ,
           ["CC","CC","FF"] ,
           ["FF","99","CC"] ,
           ["CC","FF","99"] ,
           ["99","CC","FF"] ,
           ["CC","CC","99"] ,
           ["99","CC","CC"] ,
           ["CC","99","CC"] ,
           [64, 71, 88],
           [0, 41, 132],
           [186, 0, 13]]

APR1400MAP =  [ [ True , False, False, False, False, True , True , True , True , True , True , True , False, False, False, False, True  ],
                [ False, False, False, True , True , True , True , True , True , True , True , True , True , True , False, False, False ],
                [ False, False, True , True , True , True , True , True , True , True , True , True , True , True , True , False, False ],
                [ False, True , True , True , True , True , True , True , True , True , True , True , True , True , True , True , False ],
                [ False, True , True , True , True , True , True , True , True , True , True , True , True , True , True , True , False ],
                [ True , True , True , True , True , True , True , True , True , True , True , True , True , True , True , True , True  ],
                [ True , True , True , True , True , True , True , True , True , True , True , True , True , True , True , True , True  ],
                [ True , True , True , True , True , True , True , True , True , True , True , True , True , True , True , True , True  ],
                [ True , True , True , True , True , True , True , True , True , True , True , True , True , True , True , True , True  ],
                [ True , True , True , True , True , True , True , True , True , True , True , True , True , True , True , True , True  ],
                [ True , True , True , True , True , True , True , True , True , True , True , True , True , True , True , True , True  ],
                [ True , True , True , True , True , True , True , True , True , True , True , True , True , True , True , True , True  ],
                [ False, True , True , True , True , True , True , True , True , True , True , True , True , True , True , True , False ],
                [ False, True , True , True , True , True , True , True , True , True , True , True , True , True , True , True , False ],
                [ False, False, True , True , True , True , True , True , True , True , True , True , True , True , True , False, False ],
                [ False, False, False, True , True , True , True , True , True , True , True , True , True , True , False, False, False ],
                [ False, False, False, False, False, True , True , True , True , True , True , True , False, False, False, False, False ] ]

APR1400_xPos  = [ "A" ,"B" ,"C" ,"D" ,"E" ,"F" ,"G" ,"H" ,"J" ,"K" ,"L" ,"M" ,"N" ,"P" ,"R" ,"S","T" ]
APR1400_yPos  = [ "01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17"]
APR1400_yPos2 = [  "1", "2", "3", "4", "5", "6", "7", "8", "9","10","11","12","13","14","15","16","17"]

OPR1000MAP =  [ [ False , False, False, False, False, True , True , True , True , True , False, False, False, False, False  ],
                [ False, False, False, True , True , True , True , True , True , True , True , True , False, False, False ],
                [ False, False, True , True , True , True , True , True , True , True , True , True , True , False, False ],
                [ False, True , True , True , True , True , True , True , True , True , True , True , True , True , False ],
                [ False, True , True , True , True , True , True , True , True , True , True , True , True , True , False ],
                [ True , True , True , True , True , True , True , True , True , True , True , True , True , True , True  ],
                [ True , True , True , True , True , True , True , True , True , True , True , True , True , True , True  ],
                [ True , True , True , True , True , True , True , True , True , True , True , True , True , True , True  ],
                [ True , True , True , True , True , True , True , True , True , True , True , True , True , True , True  ],
                [ True , True , True , True , True , True , True , True , True , True , True , True , True , True , True  ],
                [ False, True , True , True , True , True , True , True , True , True , True , True , True , True , False ],
                [ False, True , True , True , True , True , True , True , True , True , True , True , True , True , False ],
                [ False, False, True , True , True , True , True , True , True , True , True , True , True , False, False ],
                [ False, False, False, True , True , True , True , True , True , True , True , True , False, False, False ],
                [ False, False, False, False, False, True , True , True , True , True , False, False, False, False, True ] ]



OPR1000MAP_INFO =  [ [ True , False, False, False, False, False, False, False, False, False, False, False, False, False, True  ],
                     [ False, False, False, False, False, False, False, False, False, False, False, False, False, False, False ],
                     [ False, False, False, False, False, False, False, False, False, False, False, False, False, False, False ],
                     [ False, False, False, False, False, False, False, False, False, False, False, False, False, False, False ],
                     [ False, False, False, False, False, False, False, False, False, False, False, False, False, False, False ],
                     [ False, False, False, False, False, False, False, False, False, False, False, False, False, False, False ],
                     [ False, False, False, False, False, False, False, False, False, False, False, False, False, False, False ],
                     [ False, False, False, False, False, False, False, False, False, False, False, False, False, False, False ],
                     [ False, False, False, False, False, False, False, False, False, False, False, False, False, False, False ],
                     [ False, False, False, False, False, False, False, False, False, False, False, False, False, False, False ],
                     [ False, False, False, False, False, False, False, False, False, False, False, False, False, False, False ],
                     [ False, False, False, False, False, False, False, False, False, False, False, False, False, False, False ],
                     [ False, False, False, False, False, False, False, False, False, False, False, False, False, False, False ],
                     [ False, False, False, False, False, False, False, False, False, False, False, False, False, False, False ],
                     [ False, False, False, False, False, False, False, False, False, False, False, False, False, False, True ] ]

OPR1000MAP_RADIAL =  [ [ False, False, False, False, False, True , True , True , True , True , False, False, False, False, False ],
                       [ False, False, False, True , True , True , True , True , True , True , True , True , False, False, False ],
                       [ False, False, True , True , True , True , True , True , True , True , True , True , True , False, False ],
                       [ False, True , True , True , True , True , True , True , True , True , True , True , True , True , False ],
                       [ False, True , True , True , True , True , True , True , True , True , True , True , True , True , False ],
                       [ True , True , True , True , True , True , True , True , True , True , True , True , True , True , True  ],
                       [ True , True , True , True , True , True , True , True , True , True , True , True , True , True , True  ],
                       [ True , True , True , True , True , True , True , True , True , True , True , True , True , True , True  ],
                       [ True , True , True , True , True , True , True , True , True , True , True , True , True , True , True  ],
                       [ True , True , True , True , True , True , True , True , True , True , True , True , True , True , True  ],
                       [ False, True , True , True , True , True , True , True , True , True , True , True , True , True , False ],
                       [ False, True , True , True , True , True , True , True , True , True , True , True , True , True , False ],
                       [ False, False, True , True , True , True , True , True , True , True , True , True , True , False, False ],
                       [ False, False, False, True , True , True , True , True , True , True , True , True , False, False, False ],
                       [ False, False, False, False, False, True , True , True , True , True , False, False, False, False, False ] ]

APR1000MAP_RADIAL =  [ [ False, False, False, False, False, True , True , True , True , True , True , True , False, False, False, False, False ],
                       [ False, False, False, True , True , True , True , True , True , True , True , True , True , True , False, False, False ],
                       [ False, False, True , True , True , True , True , True , True , True , True , True , True , True , True , False, False ],
                       [ False, True , True , True , True , True , True , True , True , True , True , True , True , True , True , True , False ],
                       [ False, True , True , True , True , True , True , True , True , True , True , True , True , True , True , True , False ],
                       [ True , True , True , True , True , True , True , True , True , True , True , True , True , True , True , True , True  ],
                       [ True , True , True , True , True , True , True , True , True , True , True , True , True , True , True , True , True  ],
                       [ True , True , True , True , True , True , True , True , True , True , True , True , True , True , True , True , True  ],
                       [ True , True , True , True , True , True , True , True , True , True , True , True , True , True , True , True , True  ],
                       [ True , True , True , True , True , True , True , True , True , True , True , True , True , True , True , True , True  ],
                       [ True , True , True , True , True , True , True , True , True , True , True , True , True , True , True , True , True  ],
                       [ True , True , True , True , True , True , True , True , True , True , True , True , True , True , True , True , True  ],
                       [ False, True , True , True , True , True , True , True , True , True , True , True , True , True , True , True , False ],
                       [ False, True , True , True , True , True , True , True , True , True , True , True , True , True , True , True , False ],
                       [ False, False, True , True , True , True , True , True , True , True , True , True , True , True , True , False, False ],
                       [ False, False, False, True , True , True , True , True , True , True , True , True , True , True , False, False, False ],
                       [ False, False, False, False, False, True , True , True , True , True , True , True , False, False, False, False, False ] ]

OPR1000MAP_RADIAL_QUART =  [
                [ True , True , True , True , True , True , True , True  ],
                [ True , True , True , True , True , True , True , True  ],
                [ True , True , True , True , True , True , True , True  ],
                [ True , True , True , True , True , True , True , False ],
                [ True , True , True , True , True , True , True , False ],
                [ True , True , True , True , True , True , False, False ],
                [ True , True , True , True , True , False, False , False ],
                [ True , True , True , False, False, False, False, True ] ]

APR1400MAP_RADIAL_QUART =  [
                [ True , True , True , True , True , True , True , True , True  ],
                [ True , True , True , True , True , True , True , True , True  ],
                [ True , True , True , True , True , True , True , True , True  ],
                [ True , True , True , True , True , True , True , True , True  ],
                [ True , True , True , True , True , True , True , True , False ],
                [ True , True , True , True , True , True , True , True , False ],
                [ True , True , True , True , True , True , True , False, False ],
                [ True , True , True , True , True , True , False, False, False ],
                [ True , True , True , True , False, False, False, False, False ] ]


OPR1000_QUAR_TO_FULL = [[0,0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8],
                        [1,0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8],
                        [2,0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8],
                        [3,0], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8],
                        [4,0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7],
                        [5,0], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7],
                        [6,0], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6],
                        [7,0], [7, 1], [7, 2], [7, 3], [7, 4], [7, 5],
                        [8,0], [8, 1], [8, 2], [8, 3], ]
OPR1000_MIDDLE = 8

OPR1000_CR_ID = [    [ ""     , ""     , ""     , ""     , ""     , ""     , "R122" , ""      , "R111" , ""      , ""    , ""     , ""     , ""      , ""     ],
                     [ ""     , ""     , ""     , ""     , ""     , "R222" , ""     , "R311" , ""      , "R211" , ""      , ""    , ""     , ""      , ""     ],
                     [ ""     , ""     , "R322" , ""     , "B23"  , ""     , "B24"  , ""      , "B11"  , ""      , "B12" , ""     , "R312" , ""      , ""     ],
                     [ ""     , ""     , ""     , "R421" , ""     , "P22"  , ""     , "R511"  , ""     , "P11"   , ""    , "R411" , ""     , ""      , ""     ],
                     [ ""     , ""     , "B22"  , ""     , "A22"  , ""     , "A23"  , ""      , "A11"  , ""      , "A12" , ""     , "B13"  , ""      , ""     ],
                     [ ""     , "R221" , ""     , "P21"  , ""     , ""     , ""     , ""      , ""     , ""      , ""    , "P12"  , ""     , "R212"  , ""     ],
                     [ "R121" , ""     , "B21"  , ""     , "A21"  , ""     , "R122" , ""      , "R121" , ""      , "A13" , ""     , "B14"  , ""      , "R112" ],
                     [ ""     , "R321" , ""     , "R521" , ""     , ""     , ""     , "R440"  , ""     , ""      , ""    , "R541" , ""     , "R341"  , ""     ],
                     [ "R132" , ""     , "B34"  , ""     , "A33"  , ""     , "R123" , ""      , "R124" , ""      , "A41" , ""     , "B41"  , ""      , "R141" ],
                     [ ""     , "R232" , ""     , "P32"  , ""     , ""     , ""     , ""      , ""     , ""      , ""    , "P41"  , ""     , "R241"  , ""     ],
                     [ ""     , ""     , "B33"  , ""     , "A32"  , ""     , "A31"  , ""      , "A43"  , ""      , "A42" , ""     , "B42"  , ""      , ""     ],
                     [ ""     , ""     , ""     , "R431" , ""     , "P31"  , ""     , "R531"  , ""     , "P42"   , ""    , "R441" , ""     , ""      , ""     ],
                     [ ""     , ""     , "R332" , ""     , "B32"  , ""     , "B31"  , ""      , "B44"  , ""      , "B43" , ""     , "R342" , ""      , ""     ],
                     [ ""     , ""     , ""     , ""     , ""     , "R231" , ""     , "R331"  , ""     , "R242"  , ""    , ""     , ""     , ""      , ""     ],
                     [ ""     , ""     , ""     , ""     , ""     , ""     , "R131" , ""      , "R142" , ""      , ""    , ""     , ""     , ""      , ""     ]  ]

OPR1000MAP_BP =  [ [ "", "" , "" , "" , "" , "" , "1", "" , "1", "" , "" , "" , "" , "", "" ],
                   [ "" , "" , "" , "S", "" , "2", "" , "3", "" , "2", "" , "S", "" , "" , ""  ],
                   [ "" , "" , "" , "" , "B", "" , "B", "" , "B", "" , "B", "" , "" , "" , ""  ],
                   [ "" , "S", "" , "" , "" , "P", "" , "5", "" , "P", "" , "" , "" , "S", ""  ],
                   [ "" , "" , "B", "" , "A", "" , "A", "" , "A", "" , "A", "" , "B", "" , ""  ],
                   [ "" , "2", "" , "P", "" , "" , "" , "" , "" , "" , "" , "P", "" , "2", ""  ],
                   [ "1", "" , "B", "" , "A", "" , "1", "" , "1", "" , "A", "" , "B", "" , "1" ],
                   [ "" , "3", "" , "5", "" , "" , "" , "" , "" , "" , "" , "5", "" , "3", ""  ],
                   [ "1", "" , "B", "" , "A", "" , "1", "" , "1", "" , "A", "" , "B", "" , "1" ],
                   [ "" , "2", "" , "P", "" , "" , "" , "" , "" , "" , "" , "P", "" , "2", ""  ],
                   [ "" , "" , "B", "" , "A", "" , "A", "" , "A", "" , "A", "" , "B", "" , ""  ],
                   [ "" , "S", "" , "" , "" , "P", "" , "5", "" , "P", "" , "" , "" , "S", ""  ],
                   [ "" , "" , "" , "" , "B", "" , "B", "" , "B", "" , "B", "" , "" , "" , ""  ],
                   [ "" , "" , "" , "S", "" , "2", "" , "3", "" , "2", "" , "S", "" , "" , ""  ],
                   [ "" , "" , "" , "" , "" , "" , "1", "" , "1", "" , "" , "" , "" , "" , "Stuck\nRod"  ] ]

APR1400MAP_BP =  [ [ "Box No.\nBank Type\nSub-Group\nInsertion %",  "" , "" , "" , "" , "" , "S", "" , "3", "" , "S", "" , "" , "" , "" , "" , "Stuck\nRod"  ],
                   [ "" , "" , "" , "" , "" , "A", "" , "B", "" , "B", "" , "A", "" , "" , "" , "" , ""  ],
                   [ "" , "" , "4", "" , "2", "" , "P", "" , "5", "" , "P", "" , "2", "" , "4", "" , ""  ],
                   [ "" , "" , "" , "B", "" , "B", "" , "1", "" , "1", "" , "B", "" , "B", "" , "" , ""  ],
                   [ "" , "" , "2", "" , "4", "" , "" , "" , "3", "" , "" , "" , "4", "" , "2", "" , ""  ],
                   [ "" , "A", "" , "B", "" , "P", "" , "A", "" , "A", "" , "P", "" , "B", "" , "A", ""  ],
                   [ "S", "" , "P", "" , "" , "" , "3", "" , "" , "" , "3", "" , "" , "" , "P", "" , "S" ],
                   [ "" , "B", "" , "1", "" , "A", "" , "2", "" , "2", "" , "A", "" , "1", "" , "B", ""  ],
                   [ "3", "" , "5", "" , "3", "" , "" , "" , "5", "" , "" , "" , "3", "" , "5", "" , "3" ],
                   [ "" , "B", "" , "1", "" , "A", "" , "2", "" , "2", "" , "A", "" , "1", "" , "B", ""  ],
                   [ "S", "" , "P", "" , "" , "" , "3", "" , "" , "" , "3", "" , "" , "" , "P", "" , "S" ],
                   [ "" , "A", "" , "B", "" , "P", "" , "A", "" , "A", "" , "P", "" , "B", "" , "A", ""  ],
                   [ "" , "" , "2", "" , "4", "" , "" , "" , "3", "" , "" , "" , "4", "" , "2", "" , ""  ],
                   [ "" , "" , "" , "B", "" , "B", "" , "1", "" , "1", "" , "B", "" , "B", "" , "" , ""  ],
                   [ "" , "" , "4", "" , "2", "" , "P", "" , "5", "" , "P", "" , "2", "" , "4", "" , ""  ],
                   [ "" , "" , "" , "" , "" , "A", "" , "B", "" , "B", "" , "A", "" , "" , "" , "" , ""  ],
                   [ "",  "" , "" , "" , "" , "" , "S", "" , "3", "" , "S", "" , "" , "" , "" , "" , ""  ] ]

OPR1000_xPos  = [  "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "R" ]
OPR1000_yPos  = [ "01","02","03","04","05","06","07","08","09","10","11","12","13","14","15" ]
OPR1000_yPos2 = [  "1", "2", "3", "4", "5", "6", "7", "8", "9","10","11","12","13","14","15" ]

OPR1000_xPos_Quart  = ["H", "J", "K", "L", "M", "N", "P", "R" ]
APR1400_xPos_Quart  = ["J", "K", "L", "M", "N", "P", "R", "S", "T" ]
OPR1000_yPos_Quart  = ["08","09","10","11","12","13","14","15" ]
APR1400_yPos_Quart  = ["09","10","11","12","13","14","15","16","17" ]

RGB_ORANGE = [ 0xff, 0xA0, 0x00]
RGB_GREEN  = [ 0xBC, 0xFD, 0x60]
RGB_BLUE   = [ 0x66, 0x99, 0xCC]
RGB_PURPLE = [ 0xff, 0x33, 0xff]
RGB_GREY01 = [ 0xCC, 0xCC, 0xCC]
RGB_RED = [186, 0, 13]

CalcOpt_SDM = 0
CalcOpt_ECP = 1
CalcOpt_ASI = 2
CalcOpt_ASI_RESTART = 3
CalcOpt_RO = 4
CalcOpt_RO_RESTART = 5
CalcOpt_RECENT = 6
CalcOpt_HEIGHT = 7
CalcOpt_KILL = 8
CalcOpt_CECOR = 9
CalcOpt_INIT = 10

select_none = 0
select_snapshot = 1
select_NDR = 2
select_Boron = 3
select_RodPos = 4

RodPosUnit_percent = 0
RodPosUnit_cm   = 1
RodPosUnit_inch = 2

convertRodPosUnit = [[ 1.0, 3.81, 1.50 ],[ 1.00 / 3.81, 1.0 , 1.50 / 3.81], [1.00 / 1.50,3.81 / 1.50, 1.0 ]]

tableDefaultColumnNum = 6
tableDecayColumnNum = 1
#tableDefaultTextSize = 9
tableDefaultTextSize = 12
#tableMinimumTextSize = 8
tableMinimumTextSize = 11

styleSheet_Table = \
    u"QTableWidget {	\n"\
"	background-color: rgb(38, 44, 53);\n"\
"	padding: 10px;\n"\
"	border-radius: 15px;\n"\
"	gridline-color: rgb(44, 49, 60);\n"\
"	border-bottom: 1px solid rgb(44, 49, 60);\n"\
"}\n"\
"QTableWidget::item{\n"\
"	border-color: rgb(44, 49, 60);\n"\
"	padding-left: 5px;\n"\
"	padding-right: 5px;\n"\
"	gridline-color: rgb(44, 49, 60);\n"\
"}\n"\
"QTableWidget::item:selected{\n"\
"	background-color: rgb(85, 170, 255);\n"\
"}\n"\
"QScrollBar:horizontal {\n"\
"    border: none;\n"\
"    background: rgb(52, 59, 72);\n"\
"    height: 14px;\n"\
"    margin: 0px 21px 0 21px;\n"\
"	border-radius: 0px;\n"\
"}\n"\
"\n"\
"QScrollBar:handle:horizontal {\n"\
"    background: rgb(91, 101, 124);\n"\
"\n"\
"}\n"\
"\n"\
" QScrollBar:vertical {\n"\
"	border: none;\n"\
"    background: rgb(52, 59, 72);\n"\
"    width: 14px;\n"\
"    margin: 21px 0 21px 0;\n"\
"	border-radius: 0px;\n"\
" }\n"\
"QHeaderView::section{\n"\
"	Background-color: rgb(39, 44, 54);\n"\
"	max-width: 30px;\n"\
"	border: 1px solid rgb(44, 49, 60);\n"\
"	border-style: none;\n"\
"    border-bottom: 1px solid rgb(44, 49, 60);\n"\
"    border-right: 1px solid rgb(44, 49, 60);\n"\
"}\n"\
"QTableWidget::horizontalHeader {	\n"\
"	background-color: rgb(190, 190, 190);\n"\
"}\n"\
"\n"\
"QHeaderView::section:horizontal\n"\
"{\n"\
"	background-color: rgb(51, 59, 70);\n"\
"	padding: 3px;\n"\
"	border-top-left-radius: 7px;\n"\
"    border-top-right-radius: 7px;\n"\
"	color: rgb(167,	174, 183);\n"\
"}\n"\
"QHeaderView::section:vertical\n"\
"{\n"\
"    border: 1px solid rgb(44, 49, 60);\n"\
"}\n"\
"\n"\
"QScrollBar:handle:vertical {\n"\
"    background: rgb(91, 101, 124);\n"\
"\n"\
"}\n"\
"\n"\
"QTableCornerButton::section{ \n"\
"\n"\
"	background-color: rgb(38, 44, 53);\n"\
"}\n"\
""
styleSheet_Table_DoubleSpinBox = \
    u"QPushButton {\n"\
"	border: 2px solid rgb(52, 59, 72);\n"\
"	border-radius: 5px;\n"\
"	background-color: rgb(52, 59, 72);\n"\
"}\n"\
"QPushButton:hover {\n"\
"	background-color: rgb(57, 65, 79);\n"\
"	border: 2px solid rgb(61, 70, 86);\n"\
"}\n"\
"QPushButton:pressed {	\n"\
"	background-color: rgb(67, 77, 93);\n"\
"	border: 2px solid rgb(43, 50, 61);\n"\
"}\n"\
""

asi_i_time = 0
asi_i_power = 1
asi_i_burnup = 2
asi_i_keff = 3
asi_i_asi = 4
asi_i_boron = 5
asi_i_fr = 6
asi_i_fxy = 7
asi_i_fq = 8
asi_i_bp = 9
asi_i_b5 = 10
asi_i_b4 = 11
asi_i_b3 = 12
asi_i_p1d = 13
asi_i_p2d = 14

asi_o_asi = 0
asi_o_boron = 1
asi_o_fr = 2
asi_o_fxy = 3
asi_o_fq = 4
asi_o_bp = 5
asi_o_b5 = 6
asi_o_b4 = 7
asi_o_b3 = 8
asi_o_p1d = 9
asi_o_p2d = 10
asi_o_time = 11
asi_o_power= 12
asi_o_reactivity = 13



styleSheet_Run = """QPushButton {
	border: 2px solid rgb(52, 59, 72);
	border-radius: 5px;
	background-color: rgb(85, 170, 255);
}
QPushButton:hover {
	background-color: rgb(72, 144, 216);
	border: 2px solid rgb(61, 70, 86);
}
QPushButton:pressed {	
	background-color: rgb(52, 59, 72);
	background-color: rgb(85, 170, 255);
}"""


styleSheet_Create_Scenarios = """QPushButton {
	border: 2px solid rgb(52, 59, 72);
	border-radius: 5px;
	background-color: rgb(255, 170, 0);
}
QPushButton:hover {
	background-color: rgb(216, 144, 0);
	border: 2px solid rgb(61, 70, 86);
}
QPushButton:pressed {	
	background-color: rgb(52, 59, 72);
	background-color: rgb(255, 170, 0);
}"""


styleSheet_Delete_Scenarios = """QPushButton {
	border: 2px solid rgb(211, 90, 92);
	border-radius: 5px;
	background-color: rgb(211, 90, 92);
}
QPushButton:hover {
	background-color: rgb(200, 80, 82);
	background-color: rgb(200, 80, 82);
}
QPushButton:pressed {	
	background-color: rgb(211, 90, 92);
	background-color: rgb(211, 90, 92);
}"""

styleSheet_Table_Normal = """QPushButton {
	border: 2px solid rgb(52, 59, 72);
	border-radius: 5px;
	background-color: rgb(52, 59, 72);
}
QPushButton:hover {
	background-color: rgb(57, 65, 79);
	border: 2px solid rgb(57, 65, 79);
}
QPushButton:pressed {	
	background-color: rgb(67, 77, 93);
	background-color: rgb(67, 77, 93);
}"""


styleSheet_Message_Button = """
QPushButton {
	border: 2px solid rgb(52, 59, 72);
	border-radius: 5px;
	background-color: rgb(52, 59, 72);
}
QPushButton:hover {
	background-color: rgb(57, 65, 79);
	border: 2px solid rgb(57, 65, 79);
}
QPushButton:pressed {	
	background-color: rgb(67, 77, 93);
	border: 2px solid rgb(67, 77, 93);
}"""

styleSheet_Message_Label = """
QMessageBox QLabel {
    font-size:16px;
}
"""


# Global Constants

sdm_astra_bias = 0.0
sdm_astra_uncertainty = 6.0
sdm_required_sdm = 6.5

inlet_temperature = 295.8333

_INPUT_TYPE_USER_       = 0
_INPUT_TYPE_SNAPSHOT_   = 1
_INPUT_TYPE_FILE_CSV_   = 2
_INPUT_TYPE_FILE_EXCEL_ = 3