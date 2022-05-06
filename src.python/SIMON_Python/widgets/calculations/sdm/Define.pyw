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
           ["CC","99","CC"] ]

APR1400MAP =  [ [ False, False, False, False, False, True , True , True , True , True , True , True , False, False, False, False, False ],
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

APR1400_xPos  = [ "A" ,"B" ,"C" ,"D" ,"E" ,"F" ,"G" ,"H" ,"J" ,"K" ,"L" ,"M" ,"N" ,"P" ,"R" ,"S" ,"T" ]
APR1400_yPos  = [ "01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17"]
APR1400_yPos2 = [  "1", "2", "3", "4", "5", "6", "7", "8", "9","10","11","12","13","14","15","16","17"]

OPR1000MAP =  [ [ False, False, False, False, False, True , True , True , True , True , False, False, False, False, False ],
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

OPR1000_xPos  = [ "A" ,"B" ,"C" ,"D" ,"E" ,"F" ,"G" ,"H" ,"J" ,"K" ,"L" ,"M" ,"N" ,"P" ,"R"  ]
OPR1000_yPos  = [ "01","02","03","04","05","06","07","08","09","10","11","12","13","14","15" ]
OPR1000_yPos2 = [  "1", "2", "3", "4", "5", "6", "7", "8", "9","10","11","12","13","14","15" ]