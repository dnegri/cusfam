from cusfam import *
import utils as ut

class Simon:

    ASI_SEARCH_FAILED = -1.0
    ASI_SEARCH_ROD_INSERTION = -1
    ASI_SEARCH_ROD_WITHDRAWL = 1

    ERROR_NO = 0
    ERROR_REACH_PDIL = 10001
    ERROR_REACH_BOTTOM = 10002
    ERROR_REACH_TOP = 10003
    ERROR_REACH_BOTTOM = 10004

    #AO_EPS = 0.005
    AO_EPS = 0.001

    restart_p_position = [381*(.5),381*(.55),
                          381*(.60), 381*(.65),
                          381*(.70),381*(.75),
                          381*(.80), 381*(.85),
                          381*(.90), 381*(.95),
                          381*(.98), 381*(1.00)]

    restart_p_power    = [0, 15, 21, 33, 48, 63, 72, 81, 87, 83, 99, 100]


    shutdown_p_position = [381*(1.00),381*(.89),
                          381*(.82), 381*(.73),
                          381*(.64),381*(.51),
                          381*(.50), ]

    shutdown_p_power    = [100, 94, 91, 88, 85, 82, 0]
    shutdown_p_power    = [100, 80, 75, 70, 65, 60, 0]



    def __init__(self, file_smg, file_tset, file_ff, file_rst):
        self.s = init(file_smg, file_tset, file_ff)
        self.g = SimonGeometry()
        getGeometry(self.s, self.g)
        self.g.core_height = sum(self.g.hz[self.g.kbc:self.g.kec])
        self.file_rst = file_rst

    def setBurnupPoints(self, burnup_points):
        setBurnupPoints(self.s, burnup_points)

    def setBurnup(self, burnup):
        setBurnup(self.s, self.file_rst, burnup)

    def calculateStatic(self, std_option):
        calcStatic(self.s, std_option)

    def calculatePinPower(self):
        calcPinPower(self.s)

    def deplete(self, xe_option, sm_option, del_burn):
        deplete(self.s, xe_option, sm_option, del_burn)

    def depleteByTime(self, xe_option, sm_option, del_time):
        depleteByTime(self.s, xe_option, sm_option, del_time)

    def depleteXeSm(self, xe_option, sm_option, tsec):
        depleteXeSm(self.s, xe_option, sm_option, tsec)

    def setRodPosition1(self, rodid, position):
        setRodPosition(self.s, rodid, position)

    def setRodPosition(self, rodids, overlaps, position):
        for i in range(len(rodids)) :
            if position + overlaps[i] < 0.0:
                self.setRodPosition1(rodids[i], 0.0)
            elif position + overlaps[i] < self.g.core_height + 0.001:
                self.setRodPosition1(rodids[i], position + overlaps[i])

    def getResult(self, result):
        getResult(self.s, result)

    def getShutDownMargin(self, std_option, rodids, overlaps, pdil, rod_pos, stuck_rods) :

        crit_bak = std_option.crit
        xe_bak = std_option.xenon
        plevel = std_option.plevel

        std_option.crit = KEFF
        std_option.xenon = XE_TR

        result = SimonResult(self.g.nxya, self.g.nz)
        self.getResult(result)
        out_eigv = result.eigv

        std_option.plevel = 0.0

        self.setRodPosition(rodids, overlaps, rod_pos)

        self.calculateStatic(std_option)
        result = SimonResult(self.g.nxya, self.g.nz)
        self.getResult(result)
        def_eigv = result.eigv

        self.setRodPosition(['R','B','A','P'], [0] * 4, 0.0)
        for stuck_rod in stuck_rods :
            self.setRodPosition1(stuck_rod, self.g.core_height)

        self.calculateStatic(std_option)
        self.getResult(result)
        in_eigv = result.eigv

        std_option.crit = crit_bak
        std_option.xenon = xe_bak
        std_option.plevel = plevel

        out_react = 1-1/out_eigv
        in_react = 1-1/in_eigv
        def_react = 1-1/def_eigv

        defect = ((def_react-out_react)*1.E5+100)
        n_1 = (out_react-in_react)*1.E5*(1+(0-6.0)/100)
        sdm = n_1-defect

        return sdm, n_1, defect, out_react*1.E5

    
    #def searchRodPosition(self, std_option, ao_target, rodids, overlaps, pdil, rod_pos, preStepPos):
    #    crit_bak = std_option.crit

    #    std_option.crit = KEFF
    #    #self.setDefinedRodPosition(rodids, pre_rod_pos)
    #    #self.setRodPosition(rodids, overlaps, rod_pos)
    #    for iRod in range(len(rodids)):
    #        self.setRodPosition1(rodids[iRod],preStepPos[iRod])
    #    self.calculateStatic(std_option)

    #    result = SimonResult()
    #    self.getResult(result)

    #    if(result.ao > ao_target):
    #        #if(preStepPos[0]>0.0):
    #        result = self.searchRodPositionDown(std_option, ao_target, rodids, overlaps, pdil, rod_pos ,preStepPos)
    #        #if(preStepPos[1]>0.0 and result.ao > ao_target):
    #        #    result = self.searchRodPositionDownR4(std_option, ao_target, rodids, overlaps, pdil, rod_pos,preStepPos)
    #        #if(preStepPos[2]>0.0 and result.ao > ao_target):
    #        #    result = self.searchRodPositionDownR3(std_option, ao_target, rodids, overlaps, pdil, rod_pos,preStepPos)
    #    else :
    #        result.rod_pos[rodids[0]] = preStepPos[0]
    #        result.rod_pos[rodids[1]] = preStepPos[1]
    #        result.rod_pos[rodids[2]] = preStepPos[2]
    #    #    result = self.searchRodPositionUp(std_option, ao_target, rodids, overlaps, pdil, rod_pos)
    #    #preStepPos[0] = result.rod_pos[rodids[0]]
    #    #preStepPos[1] = result.rod_pos[rodids[1]]
    #    #preStepPos[2] = result.rod_pos[rodids[2]]

    #    std_option.crit = CBC
    #    self.calculateStatic(std_option)
    #    self.getResult(result)

    #    std_option.crit = crit_bak

    #    return result

    def searchRodPosition(self, std_option, ao_target, rodids, overlaps, pdil, rod_pos, preStepPos):
        crit_bak = std_option.crit

        std_option.crit = KEFF
        #self.setRodPosition(rodids, overlaps, rod_pos)
        #for i in range(len(rodids)):
        #    self.setRodPosition1(rodids[i],190.5)
        for iRod in range(len(rodids)):
            self.setRodPosition1(rodids[iRod],preStepPos[iRod])
        self.calculateStatic(std_option)
        self.calculatePinPower()
        #if(self.preStepFlag==True and len(self.rodids)!=0):
        #    #self.setRodPosition(rodids, overlaps, rod_pos)
        #    for iRod in range(len(self.rodids)):
        #        self.setRodPosition1(self.rodids[iRod],self.rodpos[iRod])
        #else:
        #    self.preStepFlag=True
        #    #self.setRodPosition(rodids, overlaps, rod_pos)


        #std_option.crit = KEFF
        #self.setRodPosition(rodids, overlaps, rod_pos)
        #self.calculateStatic(std_option)


        #self.setRodPosition1(rodids[0],190.5)
        #self.setRodPosition1(rodids[1],190.5)

        result = SimonResult(self.g.nxya, self.g.nz)
        self.getResult(result)
        #for i in range(len(rodids)) :
        #    result.rod_pos[rodids[i]] = min(rod_pos + overlaps[i],self.g.core_height)
        #tmp = -result.ao
        midLength = self.g.core_height/2.0
        current_ao = result.asi*-1


        pdils = ut.getPIDL_Rods(std_option.plevel * 100, preStepPos)
        # print("pdils", pdils, preStepPos)
        if current_ao > ao_target + 0.001 :
            if abs(preStepPos[0]-midLength) > 0.001:
                result = self.searchUnitRodPositionDown(0, std_option, ao_target, rodids, overlaps, pdils[0], rod_pos,preStepPos)
                current_ao = result.asi * -1.0
                if current_ao > ao_target + 0.001:
                    result.rod_pos[rodids[0]] = preStepPos[0]
                    result.rod_pos[rodids[1]] = preStepPos[1]
                    result.rod_pos[rodids[2]] = preStepPos[2]
            elif abs(preStepPos[1]-midLength) > 0.001:
                result = self.searchUnitRodPositionDown(1, std_option, ao_target, rodids, overlaps, pdils[1], rod_pos,preStepPos)
                current_ao = result.asi * -1.0
                if current_ao > ao_target + 0.001:
                    result.rod_pos[rodids[0]] = preStepPos[0]
                    result.rod_pos[rodids[1]] = preStepPos[1]
                    result.rod_pos[rodids[2]] = preStepPos[2]
            else:
                pass
                # result = self.searchUnitRodPositionDown(2, std_option, ao_target, rodids, overlaps, pdils[2], rod_pos,preStepPos)
        else :
            result.rod_pos[rodids[0]] = preStepPos[0]
            result.rod_pos[rodids[1]] = preStepPos[1]
            result.rod_pos[rodids[2]] = preStepPos[2]

        std_option.crit = CBC
        std_option.ppm = result.ppm
        self.calculateStatic(std_option)
        self.calculatePinPower()
        self.getResult(result)
        # print(result.fxy)

        std_option.crit = crit_bak

        return result

    def search_ECP_RodPosition(self, std_option, target_ppm, rodids, overlaps, pdil, rod_pos, preStepPos):
        std_option.crit = CBC
        std_option.plevel = 0.0
        result = SimonResult(self.g.nxya, self.g.nz)
        self.calculateStatic(std_option)
        self.getResult(result)
        currentPPM = result.ppm

        #p_position = self.setRodPShutdown(std_option.plevel*100)
        p_position = 381.0
        midLength = self.g.core_height/2.0
        
        if currentPPM > target_ppm + 0.001:
            result = self.searchUnitRodPositionDown_PPM(0, std_option, target_ppm, rodids, overlaps, pdil, rod_pos,preStepPos,p_position)
            # currentPPM = result.ppm
        else:
            result = self.searchRodPositionUp_PPM(std_option, target_ppm, rodids, overlaps, pdil, preStepPos[0])
            # TODO SGH, Make Rod Position Liftup sernario

        # print("error", result.error, self.ERROR_REACH_PDIL)
        try:
            rod_store = {rodids[0]:result.rod_pos[rodids[0]],
                         rodids[1]:result.rod_pos[rodids[1]],
                         rodids[2]:result.rod_pos[rodids[2]],
                         "P":p_position,}
        except:
            rod_store = {rodids[0]:preStepPos[0],
                         rodids[1]:preStepPos[1],
                         rodids[2]:preStepPos[2],
                         "P":p_position,}

        std_option.ppm = result.ppm
        std_option.crit = CBC
        self.calculateStatic(std_option)
        self.calculatePinPower()
        self.getResult(result)

        result.rod_pos[rodids[0]] = rod_store[rodids[0]]
        result.rod_pos[rodids[1]] = rod_store[rodids[1]]
        result.rod_pos[rodids[2]] = rod_store[rodids[2]]
        result.rod_pos["P"] = p_position

        return result

    def search_ECP_RodPosition_React(self, std_option, target_ppm, rodids, overlaps, pdil, rod_pos, preStepPos):
        std_option.ppm = target_ppm
        std_option.crit = KEFF
        std_option.plevel = 0.0
        result = SimonResult(self.g.nxya, self.g.nz)
        self.calculateStatic(std_option)
        self.getResult(result)
        currentEIGV = result.eigv
        currentReact = 1-(1/currentEIGV)
        target_react = 0.0

        # p_position = self.setRodPShutdown(std_option.plevel*100)
        p_position = 381.0

        #only consider r5 and r4
        end_rod_length = 2

        if -0.00001 > currentReact or 0.00001 < currentReact:
            if currentReact > 0:
                result = self.searchRodPositionDown_React(std_option, target_react, rodids, overlaps, pdil, rod_pos ,preStepPos, end_rod_length)
                # result = self.searchUnitRodPositionDown_React(0, std_option, 0.0, rodids, overlaps, pdil, rod_pos,
                #                                             preStepPos, p_position)
                # currentPPM = result.ppm
            else:
                result = self.searchRodPositionUp_React(std_option, 0.0, rodids, overlaps, pdil, preStepPos[0])

        # print("error", result.error, self.ERROR_REACH_PDIL)
        try:
            rod_store = {rodids[0]: result.rod_pos[rodids[0]],
                         rodids[1]: result.rod_pos[rodids[1]],
                         rodids[2]: result.rod_pos[rodids[2]],
                         "P": p_position, }
        except:
            rod_store = {rodids[0]: preStepPos[0],
                         rodids[1]: preStepPos[1],
                         rodids[2]: preStepPos[2],
                         "P": p_position, }

        # std_option.ppm = result.ppm
        # std_option.crit = CBC
        # self.calculateStatic(std_option)
        # self.calculatePinPower()
        # self.getResult(result)

        result.rod_pos[rodids[0]] = rod_store[rodids[0]]
        result.rod_pos[rodids[1]] = rod_store[rodids[1]]
        result.rod_pos[rodids[2]] = rod_store[rodids[2]]
        result.rod_pos["P"] = p_position

        return result

    def searchRodPositionUp(self, std_option, ao_target, rodids, overlaps, pdil, rod_pos) :

        pos_node = 0.0
        k_node = self.g.kbc
        for k in range(self.g.kbc, self.g.kec):
            pos_node += self.g.hz[k]

            if pos_node - 0.01 > rod_pos:
                k_node = k
                break

        result = SimonResult(self.g.nxya, self.g.nz)

        k_node-=1
        pos_node -= self.g.hz[k_node]
        rod_pos = pos_node

        for k in range(k_node,self.g.kec) :
            rod_pos += self.g.hz[k]
            self.setRodPosition(rodids, overlaps, rod_pos)
            self.calculateStatic(std_option)
            self.getResult(result)
            current_ao = result.asi * -1
            if current_ao > ao_target - 0.001 :
                k_node = k
                break

            if rod_pos < pdil : 
                result.error = self.ERROR_REACH_PDIL
                return result

        rod_pos -= self.g.hz[k_node]

        hz_node = self.g.hz[k_node]
        nz_div = 5
        hz_div = hz_node / nz_div
        
        for i in range(5) :
            rod_pos += hz_div
            self.setRodPosition(rodids, overlaps, rod_pos)
            self.calculateStatic(std_option)
            self.getResult(result)

            current_ao = result.asi * -1
            if current_ao > ao_target - 0.001 :
                break

        current_ao = result.asi * -1
        if abs(current_ao - ao_target) <= 0.001 :
            result.error = self.ERROR_NO
        else :
            result.error = self.ERROR_REACH_TOP
    
        for i in range(len(rodids)) :
            result.rod_pos[rodids[i]] = min(rod_pos + overlaps[i],self.g.core_height)

        return result

    def searchRodPositionDown(self, std_option, ao_target, rodids, overlaps, pdil, rod_pos,preStepPos) :
        pos_node = 0.0
        k_node = self.g.kbc
        if(preStepPos[0]>0.001):
            rod_pos = preStepPos[0]
        elif(preStepPos[1]>0.001):
            rod_pos = preStepPos[1]
            minusOverlaps = overlaps[1]
            for idx in range(len(overlaps)):
                overlaps[idx] = overlaps[idx] - minusOverlaps
        elif(preStepPos[2]>0.001):
            rod_pos = preStepPos[2]
            minusOverlaps = overlaps[2]
            for idx in range(len(overlaps)):
                overlaps[idx] = overlaps[idx] - minusOverlaps


        for k in range(self.g.kbc, self.g.kec) :
            pos_node += self.g.hz[k]
            k_node = k
            if(pos_node - 0.01 > rod_pos) : 
                break

        result = SimonResult(self.g.nxya, self.g.nz)
        rod_pos = pos_node
        
        for k in range(k_node,self.g.kbc-1,-1) :
            self.setRodPosition(rodids, overlaps, rod_pos)
            self.calculateStatic(std_option)
            self.getResult(result)

            current_ao = result.asi * -1
            if current_ao < ao_target + 0.001 :
                k_node = k
                break

            if rod_pos < pdil : 
                result.error = self.ERROR_REACH_PDIL
                return result

            rod_pos -= self.g.hz[k]

        current_ao = result.asi * -1
        if(k==self.g.kbc+1 and current_ao > ao_target):
            return result


        rod_pos += self.g.hz[k_node]

        hz_node = self.g.hz[k_node]
        nz_div = 5
        hz_div = hz_node / nz_div
        
        for i in range(5) :
            rod_pos -= hz_div
            self.setRodPosition(rodids, overlaps, rod_pos)
            self.calculateStatic(std_option)
            self.getResult(result)

            current_ao = result.asi * -1
            if current_ao < ao_target + 0.001 :
                break

        current_ao = result.asi * -1
        if abs(current_ao - ao_target) <= 0.001 :
            result.error = self.ERROR_NO
        else :
            result.error = self.ERROR_REACH_TOP
    
        for i in range(len(rodids)) :
            currentPos01 = max(rod_pos + overlaps[i], 0.0)
            currentPos   = min(currentPos01,self.g.core_height)
            #result.rod_pos[rodids[i]] = min(rod_pos + overlaps[i],self.g.core_height)
            result.rod_pos[rodids[i]] = min(currentPos,self.g.core_height)
            preStepPos[i] = currentPos

        return result


    def setDefinedRodPosition(self, rodids, position) :
        for i in range(len(rodids)) :
            self.setRodPosition1(rodids[i], position[i]) #*self.g.core_height*0.01)

    def searchASI(self, std_option, rodids, rod_pos) :
        crit_bak = std_option.crit

        std_option.crit = KEFF
        self.calculateStatic(std_option)

        result = SimonResult(self.g.nxya, self.g.nz)

        for i in range(len(rodids)):
            result.rod_pos[rodids[i]] = rod_pos[i] # * self.g.core_height / 100.0

        self.setDefinedRodPosition(rodids, rod_pos)
        std_option.crit = CBC
        self.calculateStatic(std_option)
        self.getResult(result)
        
        return result


    def searchRodPositionDown_R5(self, std_option, ao_target, rodids, overlaps, pdil, rod_pos,preStepPos) :
        pos_node = 0.0
        k_node = self.g.kbc
        for k in range(self.g.kbc, self.g.kec) :
            pos_node += self.g.hz[k]
            k_node = k
            if(pos_node - 0.01 > preStepPos[0]) :
                break

        result = SimonResult(self.g.nxya, self.g.nz)
        preStepPos[0] = pos_node
        
        kmc = int((self.g.kbc + self.g.kec)/2)-1
        
        # 1. Find Rod Position Based on Main Axial Mesh
        for k in range(k_node,kmc,-1) :
            #rod_pos -= self.g.hz[k]
            preStepPos[0] -= self.g.hz[k]
            for iRod in range(len(rodids)):
                self.setRodPosition1(rodids[iRod],preStepPos[iRod])
            #self.setRodPosition(rodids, overlaps, rod_pos)
            self.calculateStatic(std_option)
            self.getResult(result)

            # 1-1.
            current_ao = result.asi * -1
            if current_ao < ao_target + 0.001 :
                k_node = k
                break

            if preStepPos[0] < pdil : 
                result.error = self.ERROR_REACH_PDIL
                return result

            #rod_pos -= self.g.hz[k]

        # 2-1. If AO result didn't match ao_target, return Mid Position
        current_ao = result.asi * -1
        if(current_ao > ao_target + 0.001 and k==kmc+1):
            return result

        # 2-2. If Calculation AO result match ao_target,
        # Divide Main Axial Grid by 5, and Recalculate for accurate AO Target
        preStepPos[0] += self.g.hz[k_node]
        #rod_pos += self.g.hz[k_node]

        hz_node = self.g.hz[k_node]
        nz_div = 5
        hz_div = hz_node / nz_div
        
        for i in range(5) :
            preStepPos[0] -= hz_div
            for iRod in range(len(rodids)):
                self.setRodPosition1(rodids[iRod],preStepPos[iRod])
            #self.setRodPosition(rodids, overlaps, rod_pos)
            self.calculateStatic(std_option)
            self.getResult(result)

            current_ao = result.asi * -1
            if current_ao < ao_target + 0.001 :
                break

        # 3. Error Message
        current_ao = result.asi * -1
        if abs(current_ao - ao_target) <= 0.001 :
            result.error = self.ERROR_NO
        #else :
        #    result.error = self.ERROR_REACH_TOP
        

        result.rod_pos[rodids[0]] = preStepPos[0]
        result.rod_pos[rodids[1]] = preStepPos[1]
        result.rod_pos[rodids[2]] = preStepPos[2]
        #for i in range(len(rodids)) :
            #result.rod_pos[rodids[i]] = min(rod_pos + overlaps[i],self.g.core_height)

        return result

    def searchRodPositionDown_R4(self, std_option, ao_target, rodids, overlaps, pdil, rod_pos,preStepPos) :
        pos_node = 0.0
        k_node = self.g.kbc
        for k in range(self.g.kbc, self.g.kec) :
            pos_node += self.g.hz[k]
            k_node = k
            if(pos_node - 0.01 > preStepPos[1]) :
                break

        result = SimonResult(self.g.nxya, self.g.nz)
        preStepPos[1] = pos_node
        
        kmc = int((self.g.kbc + self.g.kec)/2)-1
        
        # 1. Find Rod Position Based on Main Axial Mesh
        for k in range(k_node,kmc,-1) :
            #rod_pos -= self.g.hz[k]
            preStepPos[1] -= self.g.hz[k]
            for iRod in range(len(rodids)):
                self.setRodPosition1(rodids[iRod],preStepPos[iRod])
            #self.setRodPosition(rodids, overlaps, rod_pos)
            self.calculateStatic(std_option)
            self.getResult(result)

            # 1-1.

            current_ao = result.asi * -1
            if current_ao < ao_target + 0.001 :
                k_node = k
                break

            if preStepPos[1] < pdil : 
                result.error = self.ERROR_REACH_PDIL
                return result

            #rod_pos -= self.g.hz[k]

        # 2-1. If AO result didn't match ao_target, return Mid Position
        current_ao = result.asi * -1
        if(current_ao < ao_target + 0.001 and k==kmc+1):
            return result

        # 2-2. If Calculation AO result match ao_target,
        # Divide Main Axial Grid by 5, and Recalculate for accurate AO Target
        preStepPos[1] += self.g.hz[k_node]
        #rod_pos += self.g.hz[k_node]

        hz_node = self.g.hz[k_node]
        nz_div = 5
        hz_div = hz_node / nz_div
        
        for i in range(5) :
            preStepPos[1] -= hz_div
            for iRod in range(len(rodids)):
                self.setRodPosition1(rodids[iRod],preStepPos[iRod])
            #self.setRodPosition(rodids, overlaps, rod_pos)
            self.calculateStatic(std_option)
            self.getResult(result)

            current_ao = result.asi * -1
            if current_ao < ao_target + 0.001 :
                break

        # 3. Error Message
        current_ao = result.asi * -1
        if abs(current_ao - ao_target) <= 0.001 :
            result.error = self.ERROR_NO
        #else :
        #    result.error = self.ERROR_REACH_TOP
        

        result.rod_pos[rodids[0]] = preStepPos[0]
        result.rod_pos[rodids[1]] = preStepPos[1]
        result.rod_pos[rodids[2]] = preStepPos[2]
        #for i in range(len(rodids)) :
            #result.rod_pos[rodids[i]] = min(rod_pos + overlaps[i],self.g.core_height)

        return result

    def searchRodPositionDown_R3(self, std_option, ao_target, rodids, overlaps, pdil, rod_pos,preStepPos) :
        pos_node = 0.0
        k_node = self.g.kbc
        for k in range(self.g.kbc, self.g.kec) :
            pos_node += self.g.hz[k]
            k_node = k
            if(pos_node - 0.01 > preStepPos[2]) :
                break

        result = SimonResult(self.g.nxya, self.g.nz)
        preStepPos[2] = pos_node
        
        kmc = int((self.g.kbc + self.g.kec)/2)-1
        
        # 1. Find Rod Position Based on Main Axial Mesh
        for k in range(k_node,kmc,-1) :
            #rod_pos -= self.g.hz[k]
            preStepPos[2] -= self.g.hz[k]
            for iRod in range(len(rodids)):
                self.setRodPosition1(rodids[iRod],preStepPos[iRod])
            #self.setRodPosition(rodids, overlaps, rod_pos)
            self.calculateStatic(std_option)
            self.getResult(result)

            # 1-1.
            current_ao = result.asi * -1
            if current_ao < ao_target + 0.001 :
                k_node = k
                break

            if preStepPos[2] < pdil : 
                result.error = self.ERROR_REACH_PDIL
                return result

            #rod_pos -= self.g.hz[k]

        # 2-1. If AO result didn't match ao_target, return Mid Position
        current_ao = result.asi * -1
        if(current_ao < ao_target + 0.001 and k==kmc+1):
            return result

        # 2-2. If Calculation AO result match ao_target,
        # Divide Main Axial Grid by 5, and Recalculate for accurate AO Target
        preStepPos[2] += self.g.hz[k_node]
        #rod_pos += self.g.hz[k_node]

        hz_node = self.g.hz[k_node]
        nz_div = 5
        hz_div = hz_node / nz_div
        
        for i in range(5) :
            preStepPos[2] -= hz_div
            for iRod in range(len(rodids)):
                self.setRodPosition1(rodids[iRod],preStepPos[iRod])
            #self.setRodPosition(rodids, overlaps, rod_pos)
            self.calculateStatic(std_option)
            self.getResult(result)

            current_ao = result.asi * -1
            if current_ao < ao_target + 0.001 :
                break

        # 3. Error Message
        current_ao = result.asi * -1
        if abs(current_ao - ao_target) <= 0.001 :
            result.error = self.ERROR_NO
        #else :
        #    result.error = self.ERROR_REACH_TOP

        result.rod_pos[rodids[0]] = preStepPos[0]
        result.rod_pos[rodids[1]] = preStepPos[1]
        result.rod_pos[rodids[2]] = preStepPos[2]
        #for i in range(len(rodids)) :
            #result.rod_pos[rodids[i]] = min(rod_pos + overlaps[i],self.g.core_height)

        return result

    def searchUnitRodPositionDown(self, iRod, std_option, ao_target, rodids, overlaps, pdil, rod_pos, preStepPos) :
        # 1. Search Current Control Rod Position
        pos_node = 0.0
        k_node = self.g.kec

        for k in range(self.g.kbc, self.g.kec):
            pos_node += self.g.hz[k]

            k_node = k
            if pos_node - 0.01 > preStepPos[iRod]:
                break

        result = SimonResult(self.g.nxya, self.g.nz)
        result.asi = -1.0
        preStepPos[iRod] = pos_node
        kmc = int((self.g.kbc + self.g.kec)/2)-1

        roll_back_position = [preStepPos[0], preStepPos[1], preStepPos[2],]

        if iRod == 0:
            if(k_node<=kmc+2):
                preStepPos[iRod+1] = pos_node + self.g.core_height * 0.4
        
        #kmc = int((self.g.kbc + self.g.kec)/2)-1
        
        # 2. If Control Rod Position is higher then 0.6 * CoreHeight,
        #    Drop Only One Control Rod
        #    If Control Rod Position is lower then 0.6 * CoreHeight,
        #    Drop Two Control Rod
        # 2-1. Find Rod Position Based on Main Axial Mesh
        if(k_node > kmc+2):
            # Case Control Rod Position > 0.6 * CoreHeight
            # sum(self.g.hz[self.g.kbc:kmc+2] == 0.6 * CoreHeight
            # Search Limit = kmc+2 to k_node
            for k in range(k_node,kmc+2,-1) :
                #rod_pos -= self.g.hz[k]
                preStepPos[iRod] -= self.g.hz[k]
                for i in range(len(rodids)):
                    self.setRodPosition1(rodids[i],preStepPos[i])
                #self.setRodPosition(rodids, overlaps, rod_pos)
                self.calculateStatic(std_option)
                self.getResult(result)

                # 1-1.
                # print("pdil position", iRod, preStepPos[iRod], pdil)
                if preStepPos[iRod] < pdil:
                    result.error = self.ERROR_REACH_PDIL
                    # print("reached pdil")
                    for i in range(len(rodids)):
                        self.setRodPosition1(rodids[i], roll_back_position[i])
                    # self.setRodPosition(rodids, overlaps, rod_pos)
                    self.calculateStatic(std_option)
                    self.getResult(result)
                    return result

                current_ao = result.asi * -1
                if current_ao < ao_target + 0.001:
                    k_node = k
                    break

                if(k==kmc+3):
                    # If Control Rod Position Reach 228.6 (190.5 + 20.0 + 18.1)
                    # And If currentPPM steel higher then targetPPM,
                    # This Methnology control current PPM by dropping two rod
                    # For Adjust control Rod Index, count out one for k
                    k = k-1

        current_ao = result.asi * -1
        if current_ao >= ao_target + 0.001 and k>kmc and iRod == 0:
            k_node = k
            # Case Control Rod Position > 0.6 * CoreHeight
            # sum(self.g.hz[self.g.kbc:kmc+2] == 0.6 * CoreHeight
            # Search Limit = kmc+2 to k_node
            for k in range(k_node,kmc,-1):
                #rod_pos -= self.g.hz[k]
                preStepPos[iRod] -= self.g.hz[k]
                if iRod == 0:
                    preStepPos[iRod+1] -= self.g.hz[k]
                for i in range(len(rodids)):
                    self.setRodPosition1(rodids[i],preStepPos[i])
                #self.setRodPosition(rodids, overlaps, rod_pos)
                self.calculateStatic(std_option)
                self.getResult(result)

                current_ao = result.asi * -1

                if preStepPos[iRod+1] < pdil:
                    result.error = self.ERROR_REACH_PDIL
                    # print("reached pdil")
                    for i in range(len(rodids)):
                        self.setRodPosition1(rodids[i], roll_back_position[i])
                    # self.setRodPosition(rodids, overlaps, rod_pos)
                    self.calculateStatic(std_option)
                    self.getResult(result)
                    return result

                if current_ao < ao_target + 0.001 :
                    k_node = k
                    break

        current_ao = result.asi * -1
        # 2-2. If AO result didn't match ao_target, return Mid Position
        if(current_ao > ao_target + 0.001 and k==kmc+1):
            # print("return mid")
            return result

        # 3. If Calculation AO result match ao_target,
        # Divide Main Axial Grid by 5, and Recalculate for accurate AO Target
        if(k>kmc+2):
            preStepPos[iRod] += self.g.hz[k_node]
            hz_node = self.g.hz[k_node]
            nz_div = 5
            hz_div = hz_node / nz_div
        
            for i in range(5) :
                preStepPos[iRod] -= hz_div
                for i in range(len(rodids)):
                    self.setRodPosition1(rodids[i],preStepPos[i])
                #self.setRodPosition(rodids, overlaps, rod_pos)
                self.calculateStatic(std_option)
                self.getResult(result)

                current_ao = result.asi * -1
                if current_ao < ao_target + 0.001 :
                    break
        else:
            preStepPos[iRod] += self.g.hz[k_node]
            if iRod == 0:
                preStepPos[iRod+1] += self.g.hz[k_node]
            hz_node = self.g.hz[k_node]
            nz_div = 5
            hz_div = hz_node / nz_div
        
            for i in range(5) :
                preStepPos[iRod] -= hz_div
                if iRod == 0:
                    preStepPos[iRod+1] -= hz_div
                for i in range(len(rodids)):
                    self.setRodPosition1(rodids[i],preStepPos[i])
                #self.setRodPosition(rodids, overlaps, rod_pos)
                self.calculateStatic(std_option)
                self.getResult(result)

                current_ao = result.asi * -1
                if current_ao < ao_target + 0.001 :
                    break
        # 3. Error Message
        current_ao = result.asi * -1
        if abs(current_ao - ao_target) <= 0.001 :
            result.error = self.ERROR_NO
        #else :
        #    result.error = self.ERROR_REACH_TOP

        result.rod_pos[rodids[0]] = preStepPos[0]
        result.rod_pos[rodids[1]] = preStepPos[1]
        result.rod_pos[rodids[2]] = preStepPos[2]
        #for i in range(len(rodids)) :
            #result.rod_pos[rodids[i]] = min(rod_pos + overlaps[i],self.g.core_height)

        return result

    def searchUnitRodPositionDown_PPM(self, iRod, std_option, targetPPM, rodids, overlaps, pdil, rod_pos, preStepPos,p_position):
        # 1. Search Current Control Rod Position
        pos_node = 0.0
        k_node = self.g.kbc
        for k in range(self.g.kbc, self.g.kec):
            pos_node += self.g.hz[k]
            k_node = k
            if (pos_node - 0.01 > preStepPos[iRod]):
                break

        result = SimonResult(self.g.nxya, self.g.nz)
        preStepPos[iRod] = pos_node
        kmc = int((self.g.kbc + self.g.kec) / 2) - 1
        if (k_node <= kmc + 2):
            preStepPos[iRod + 1] = pos_node + self.g.core_height * 0.4

        # kmc = int((self.g.kbc + self.g.kec)/2)-1

        # 2. If Control Rod Position is higher then 0.6 * CoreHeight (kmc+2),
        #    Drop Only One Control Rod
        #    If Control Rod Position is lower then 0.6 * CoreHeight,
        #    Drop Two Control Rod
        # 2-1. Find Rod Position Based on Main Axial Mesh
        currentPPM = 0.0
        if (k_node > kmc + 2):
            # Case Control Rod Position > 0.6 * CoreHeight
            # sum(self.g.hz[self.g.kbc:kmc+2] == 0.6 * CoreHeight
            # Search Limit = kmc+2 to k_node
            for k in range(k_node, kmc + 2, -1):
                # rod_pos -= self.g.hz[k]
                preStepPos[iRod] -= self.g.hz[k]
                for i in range(len(rodids)):
                    self.setRodPosition1(rodids[i], preStepPos[i])
                self.setRodPosition1("P", p_position)
                self.calculateStatic(std_option)
                self.getResult(result)
                currentPPM = result.ppm

                if currentPPM < targetPPM + 1.0:
                    k_node = k
                    break

                if preStepPos[iRod] < pdil:
                    result.error = self.ERROR_REACH_PDIL
                    return result
                if(k==kmc+3):
                    # If Control Rod Position Reach 228.6 (190.5 + 20.0 + 18.1)
                    # And If currentPPM steel higher then targetPPM, 
                    # This Methnology control current PPM by dropping two rod
                    # For Adjust control Rod Index, count out one for k
                    k = k-1

        if (currentPPM >= targetPPM + 0.001 and k > kmc and (len(rodids) - 1 != iRod)):
            k_node = k
            # Case Control Rod Position > 0.6 * CoreHeight
            # sum(self.g.hz[self.g.kbc:kmc+2] == 0.6 * CoreHeight
            # Search Limit = kmc+2 to k_node
            for k in range(k_node, kmc, -1):
                # rod_pos -= self.g.hz[k]
                preStepPos[iRod] -= self.g.hz[k]
                preStepPos[iRod + 1] -= self.g.hz[k]
                for i in range(len(rodids)):
                    self.setRodPosition1(rodids[i], preStepPos[i])
                self.setRodPosition1("P", p_position)
                # self.setRodPosition(rodids, overlaps, rod_pos)
                self.calculateStatic(std_option)
                self.getResult(result)
                currentPPM = result.ppm
                if currentPPM < targetPPM + 1.0:
                    k_node = k
                    break

        # 2-2. If AO result didn't match ao_target, return Mid Position
        # if (currentPPM > targetPPM + 0.001 and k == kmc + 1):
        #     return result

        # 3. If Calculation AO result match ao_target,
        # Divide Main Axial Grid by 5, and Recalculate for accurate AO Target
        if (k > kmc + 2):
            lowPPM = result.ppm
            preStepPos[iRod] += self.g.hz[k_node]
            for i in range(len(rodids)):
                self.setRodPosition1(rodids[i], preStepPos[i])
            self.setRodPosition1("P", p_position)
            self.calculateStatic(std_option)
            self.getResult(result)
            highPPM = result.ppm

            hz_node = self.g.hz[k_node]
            nz_div = 5
            hz_div = hz_node / nz_div

            for i in range(5):
                preStepPos[iRod] -= hz_div
                for i in range(len(rodids)):
                    self.setRodPosition1(rodids[i], preStepPos[i])
                self.setRodPosition1("P", p_position)
                # self.setRodPosition(rodids, overlaps, rod_pos)
                self.calculateStatic(std_option)
                self.getResult(result)
                currentPPM = result.ppm
                if currentPPM < targetPPM + 1.0:
                    break
        else:
            if iRod < 2:
                preStepPos[iRod] += self.g.hz[k_node]
                preStepPos[iRod + 1] += self.g.hz[k_node]
                hz_node = self.g.hz[k_node]
                nz_div = 5
                hz_div = hz_node / nz_div

                for i in range(5):
                    preStepPos[iRod] -= hz_div
                    preStepPos[iRod + 1] -= hz_div
                    for i in range(len(rodids)):
                        self.setRodPosition1(rodids[i], preStepPos[i])
                    self.setRodPosition1("P", p_position)
                    # self.setRodPosition(rodids, overlaps, rod_pos)
                    self.calculateStatic(std_option)
                    self.getResult(result)

                    currentPPM = result.ppm
                    if currentPPM < targetPPM + 1.0:
                        break
        # 3. Error Message
        if abs(currentPPM - targetPPM) <= 1.0:
            result.error = self.ERROR_NO
        # else :
        #    result.error = self.ERROR_REACH_TOP

        result.rod_pos[rodids[0]] = preStepPos[0]
        result.rod_pos[rodids[1]] = preStepPos[1]
        result.rod_pos[rodids[2]] = preStepPos[2]
        result.rod_pos["P"] = p_position
        # for i in range(len(rodids)) :
        # result.rod_pos[rodids[i]] = min(rod_pos + overlaps[i],self.g.core_height)

        return result

    # def searchUnitRodPositionDown_React(self, iRod, std_option, targetReact, rodids, overlaps, pdil, rod_pos, preStepPos,p_position):
    #     # 1. Search Current Control Rod Position
    #     pos_node = 0.0
    #     k_node = self.g.kbc
    #     for k in range(self.g.kbc, self.g.kec):
    #         pos_node += self.g.hz[k]
    #         k_node = k
    #         if (pos_node - 0.01 > preStepPos[iRod]):
    #             break
    #
    #     result = SimonResult(self.g.nxya, self.g.nz)
    #     preStepPos[iRod] = pos_node
    #     kmc = int((self.g.kbc + self.g.kec) / 2) - 1
    #     if (k_node <= kmc + 2):
    #         preStepPos[iRod + 1] = pos_node + self.g.core_height * 0.4
    #
    #     # kmc = int((self.g.kbc + self.g.kec)/2)-1
    #
    #     # 2. If Control Rod Position is higher then 0.6 * CoreHeight (kmc+2),
    #     #    Drop Only One Control Rod
    #     #    If Control Rod Position is lower then 0.6 * CoreHeight,
    #     #    Drop Two Control Rod
    #     # 2-1. Find Rod Position Based on Main Axial Mesh
    #     currentReact = 0.0
    #     if k_node > kmc + 2:
    #         # Case Control Rod Position > 0.6 * CoreHeight
    #         # sum(self.g.hz[self.g.kbc:kmc+2] == 0.6 * CoreHeight
    #         # Search Limit = kmc+2 to k_node
    #         for k in range(k_node, kmc + 2, -1):
    #             # rod_pos -= self.g.hz[k]
    #             preStepPos[iRod] -= self.g.hz[k]
    #             for i in range(len(rodids)):
    #                 self.setRodPosition1(rodids[i], preStepPos[i])
    #             self.setRodPosition1("P", p_position)
    #             self.calculateStatic(std_option)
    #             self.getResult(result)
    #             currentReact = 1-(1/result.eigv)
    #             if currentReact < targetReact + 0.00001:
    #                 k_node = k
    #                 break
    #
    #             if preStepPos[iRod] < pdil:
    #                 result.error = self.ERROR_REACH_PDIL
    #                 return result
    #             if(k==kmc+3):
    #                 # If Control Rod Position Reach 228.6 (190.5 + 20.0 + 18.1)
    #                 # And If currentPPM steel higher then targetPPM,
    #                 # This Methnology control current PPM by dropping two rod
    #                 # For Adjust control Rod Index, count out one for k
    #                 k = k-1
    #
    #     if (currentReact >= targetReact + 0.001 and k > kmc and (len(rodids) - 1 != iRod)):
    #         k_node = k
    #         # Case Control Rod Position > 0.6 * CoreHeight
    #         # sum(self.g.hz[self.g.kbc:kmc+2] == 0.6 * CoreHeight
    #         # Search Limit = kmc+2 to k_node
    #         for k in range(k_node, kmc, -1):
    #             # rod_pos -= self.g.hz[k]
    #             preStepPos[iRod] -= self.g.hz[k]
    #             preStepPos[iRod + 1] -= self.g.hz[k]
    #             for i in range(len(rodids)):
    #                 self.setRodPosition1(rodids[i], preStepPos[i])
    #             self.setRodPosition1("P", p_position)
    #             # self.setRodPosition(rodids, overlaps, rod_pos)
    #             self.calculateStatic(std_option)
    #             self.getResult(result)
    #             currentReact = 1-(1/result.eigv)
    #             if currentReact < targetReact + 0.00001:
    #                 k_node = k
    #                 break
    #
    #     # 2-2. If AO result didn't match ao_target, return Mid Position
    #     # if (currentPPM > targetPPM + 0.001 and k == kmc + 1):
    #     #     return result
    #
    #     # 3. If Calculation AO result match ao_target,
    #     # Divide Main Axial Grid by 5, and Recalculate for accurate AO Target
    #     if (k > kmc + 2):
    #         lowPPM = result.ppm
    #         preStepPos[iRod] += self.g.hz[k_node]
    #         for i in range(len(rodids)):
    #             self.setRodPosition1(rodids[i], preStepPos[i])
    #         self.setRodPosition1("P", p_position)
    #         self.calculateStatic(std_option)
    #         self.getResult(result)
    #         highPPM = result.ppm
    #
    #         hz_node = self.g.hz[k_node]
    #         nz_div = 5
    #         hz_div = hz_node / nz_div
    #
    #         for i in range(5):
    #             preStepPos[iRod] -= hz_div
    #             for i in range(len(rodids)):
    #                 self.setRodPosition1(rodids[i], preStepPos[i])
    #             self.setRodPosition1("P", p_position)
    #             # self.setRodPosition(rodids, overlaps, rod_pos)
    #             self.calculateStatic(std_option)
    #             self.getResult(result)
    #             currentReact = 1-(1/result.eigv)
    #             if currentReact < targetReact + 0.00001:
    #                 break
    #     else:
    #         if iRod < 2:
    #             preStepPos[iRod] += self.g.hz[k_node]
    #             preStepPos[iRod + 1] += self.g.hz[k_node]
    #             hz_node = self.g.hz[k_node]
    #             nz_div = 5
    #             hz_div = hz_node / nz_div
    #
    #             for i in range(5):
    #                 preStepPos[iRod] -= hz_div
    #                 preStepPos[iRod + 1] -= hz_div
    #                 for i in range(len(rodids)):
    #                     self.setRodPosition1(rodids[i], preStepPos[i])
    #                 self.setRodPosition1("P", p_position)
    #                 # self.setRodPosition(rodids, overlaps, rod_pos)
    #                 self.calculateStatic(std_option)
    #                 self.getResult(result)
    #
    #                 currentReact = 1 - (1 / result.eigv)
    #                 if currentReact < targetReact + 0.00001:
    #                     break
    #
    #     # 3. Error Message
    #     if abs(currentReact - targetReact) <= 0.00001:
    #         result.error = self.ERROR_NO
    #     # else :
    #     #    result.error = self.ERROR_REACH_TOP
    #
    #     result.rod_pos[rodids[0]] = preStepPos[0]
    #     result.rod_pos[rodids[1]] = preStepPos[1]
    #     result.rod_pos[rodids[2]] = preStepPos[2]
    #     result.rod_pos["P"] = p_position
    #     # for i in range(len(rodids)) :
    #     # result.rod_pos[rodids[i]] = min(rod_pos + overlaps[i],self.g.core_height)
    #
    #     return result

    def searchRodPositionDown_React(self, std_option, target_react, rodids, overlaps, pdil, rod_pos, preStepPos, end_rod_length=3):
        pos_node = 0.0
        k_node = self.g.kbc

        if (preStepPos[0] > 0.001):
            rod_pos = preStepPos[0]
        elif (preStepPos[1] > 0.00 and end_rod_length > 1):
            rod_pos = preStepPos[1]
            minusOverlaps = overlaps[1]
            for idx in range(len(overlaps)):
                overlaps[idx] = overlaps[idx] - minusOverlaps
        elif (preStepPos[2] > 0.001 and end_rod_length > 2):
            rod_pos = preStepPos[2]
            minusOverlaps = overlaps[2]
            for idx in range(len(overlaps)):
                overlaps[idx] = overlaps[idx] - minusOverlaps

        for k in range(self.g.kbc, self.g.kec):
            pos_node += self.g.hz[k]
            k_node = k
            if (pos_node - 0.01 > rod_pos):
                break

        result = SimonResult(self.g.nxya, self.g.nz)
        rod_pos = pos_node

        for k in range(k_node, self.g.kbc - 1, -1):
            self.setRodPosition(rodids[:end_rod_length], overlaps[:end_rod_length], rod_pos)
            self.calculateStatic(std_option)
            self.getResult(result)

            current_react = 1 - (1 / result.eigv)
            if current_react < target_react + 0.00001:
                k_node = k
                break

            if rod_pos < pdil:
                result.error = self.ERROR_REACH_PDIL
                return result

            rod_pos -= self.g.hz[k]

        current_react = 1 - (1 / result.eigv)
        if (k == self.g.kbc + 1 and current_react > target_react):
            return result

        rod_pos += self.g.hz[k_node]

        hz_node = self.g.hz[k_node]
        nz_div = 10
        hz_div = hz_node / nz_div

        for i in range(nz_div):
            rod_pos -= hz_div
            self.setRodPosition(rodids[:end_rod_length], overlaps[:end_rod_length], rod_pos)
            self.calculateStatic(std_option)
            self.getResult(result)

            current_react = 1 - (1 / result.eigv)
            if current_react < target_react + 0.00001:
                break

        current_react = 1 - (1 / result.eigv)
        if current_react < target_react + 0.00001:
            result.error = self.ERROR_NO
        else:
            result.error = self.ERROR_REACH_TOP

        for i in range(len(rodids)):
            if i < end_rod_length:
                currentPos01 = max(rod_pos + overlaps[i], 0.0)
                currentPos = min(currentPos01, self.g.core_height)
                # result.rod_pos[rodids[i]] = min(rod_pos + overlaps[i],self.g.core_height)
                result.rod_pos[rodids[i]] = min(currentPos, self.g.core_height)
                preStepPos[i] = currentPos

        return result


    def searchRodPositionUp_PPM(self, std_option, targetReact, rodids, overlaps, pdil, rod_pos) :

        pos_node = 0.0
        k_node = self.g.kbc
        for k in range(self.g.kbc, self.g.kec) :
            pos_node += self.g.hz[k]

            k_node = k
            if(pos_node - 0.01 > rod_pos) :
                break

        result = SimonResult(self.g.nxya, self.g.nz)

        k_node-=1
        pos_node -= self.g.hz[k_node]
        rod_pos = pos_node

        for k in range(k_node,self.g.kec) :
            rod_pos += self.g.hz[k]
            self.setRodPosition(rodids, overlaps, rod_pos)
            self.calculateStatic(std_option)
            self.getResult(result)
            currentReact = 1-(1/result.eigv)
            if currentReact < targetReact + 0.00001:
                k_node = k
                break

            if rod_pos < pdil : 
                result.error = self.ERROR_REACH_PDIL
                return result

        rod_pos -= self.g.hz[k_node]

        hz_node = self.g.hz[k_node]
        nz_div = 5
        hz_div = hz_node / nz_div
        
        for i in range(5) :
            rod_pos += hz_div
            self.setRodPosition(rodids, overlaps, rod_pos)
            self.calculateStatic(std_option)
            self.getResult(result)
            currentPPM = result.ppm

            if currentPPM > targetPPM - 0.001 :
                break

        if abs(currentPPM - targetPPM) <= 0.001 :
            result.error = self.ERROR_NO
        else :
            result.error = self.ERROR_REACH_TOP
    
        for i in range(len(rodids)):
            result.rod_pos[rodids[i]] = min(rod_pos + overlaps[i],self.g.core_height)

        return result

    def searchRodPositionUp_React(self, std_option, targetReact, rodids, overlaps, pdil, rod_pos):

        pos_node = 0.0
        k_node = self.g.kbc
        for k in range(self.g.kbc, self.g.kec):
            pos_node += self.g.hz[k]

            k_node = k
            if (pos_node - 0.01 > rod_pos):
                break

        result = SimonResult(self.g.nxya, self.g.nz)

        k_node -= 1
        pos_node -= self.g.hz[k_node]
        rod_pos = pos_node

        for k in range(k_node, self.g.kec):
            rod_pos += self.g.hz[k]
            self.setRodPosition(rodids, overlaps, rod_pos)
            self.calculateStatic(std_option)
            self.getResult(result)

            currentReact = 1-(1/result.eigv)
            if currentReact > targetReact - 0.00001:
                k_node = k
                break

            if rod_pos < pdil:
                result.error = self.ERROR_REACH_PDIL
                return result

        rod_pos -= self.g.hz[k_node]

        hz_node = self.g.hz[k_node]
        nz_div = 5
        hz_div = hz_node / nz_div

        for i in range(5):
            rod_pos += hz_div
            self.setRodPosition(rodids, overlaps, rod_pos)
            self.calculateStatic(std_option)
            self.getResult(result)

            currentReact = 1-1/result.eigv
            if currentReact > targetReact - 0.00001:
                break

        if abs(currentReact - targetReact) <= 0.00001:
            result.error = self.ERROR_NO
        else:
            result.error = self.ERROR_REACH_TOP

        for i in range(len(rodids)):
            result.rod_pos[rodids[i]] = min(rod_pos + overlaps[i], self.g.core_height)

        return result

    def searchRodPositionO(self, std_option, asi_target, rodids, overlaps, pdil, rod_pos):
        crit_bak = std_option.crit

        std_option.crit = KEFF
        self.setRodPosition(rodids, overlaps, rod_pos)
        self.calculateStatic(std_option)

        result = SimonResult(self.g.nxya, self.g.nz)
        self.getResult(result)

        if result.asi < asi_target and rod_pos >= 0.50*381:
            result = self.searchRodPositionDownO(std_option, asi_target, rodids, overlaps, pdil, rod_pos)
        else:
            result = self.searchRodPositionUpO(std_option, asi_target, rodids, overlaps, pdil, rod_pos)
        #
        # pdil_position = self.getPDIL(std_option.plevel * 100)
        # if result.asi > asi_target:
        #     result = self.searchRodPositionDownO(std_option, asi_target, rodids, overlaps, pdil, rod_pos)
        # else:
        #     # result = self.searchRodPositionUpO(std_option, asi_target, rodids, overlaps, pdil, rod_pos)
        #     for i in range(len(rodids)):
        #         rod_position = rod_pos + overlaps[i]
        #         result.rod_pos[rodids[i]] = min(rod_position, self.g.core_height)
        #
        # print(std_option.plevel * 100, pdil_position, result.rod_pos)
        #
        # for i in range(len(rodids)):
        #     result.rod_pos[rodids[i]] = max(result.rod_pos[rodids[i]], pdil_position[i])

        #print("here", result.rod_pos['P'])
        std_option.crit = CBC
        self.calculateStatic(std_option)
        self.calculatePinPower()
        self.getResult(result)

        std_option.crit = crit_bak

        return result


    def searchRodPositionUpO(self, std_option, asi_target, rodids, overlaps, pdil, rod_pos):
        pos_node = 0.0
        k_node = self.g.kbc
        for k in range(self.g.kbc, self.g.kec):
            pos_node += self.g.hz[k]
            k_node = k
            if (pos_node - 0.01 > rod_pos):
                break

        result = SimonResult(self.g.nxya, self.g.nz)
        result.error = self.ERROR_NO

        pos_node -= self.g.hz[k_node]
        rod_pos = pos_node

        for k in range(k_node, self.g.kec):
            rod_pos += self.g.hz[k]

            self.setRodPosition(rodids, overlaps, rod_pos)
            self.calculateStatic(std_option)
            self.getResult(result)

            k_node = k
            if result.asi < asi_target - self.AO_EPS:
                break

            if rod_pos < pdil:
                result.error = self.ERROR_REACH_PDIL
                break

            # print(result.error, result.asi)

        if result.asi > asi_target - self.AO_EPS:
            result.error = self.ERROR_REACH_TOP

        for i in range(len(rodids)):
            result.rod_pos[rodids[i]] = min(rod_pos + overlaps[i], self.g.core_height)

        if result.error != self.ERROR_NO:
            return result

        rod_pos -= self.g.hz[k_node]

        hz_node = self.g.hz[k_node]
        nz_div = 5
        hz_div = hz_node / nz_div

        result.error = self.ERROR_REACH_TOP

        for i in range(5):
            rod_pos += hz_div
            self.setRodPosition(rodids, overlaps, rod_pos)
            self.calculateStatic(std_option)
            self.getResult(result)

            if result.asi < asi_target + self.AO_EPS:
                result.error = self.ERROR_NO
                break

        for i in range(len(rodids)):
            result.rod_pos[rodids[i]] = min(rod_pos + overlaps[i], self.g.core_height)

        return result

    def searchRodPositionDownO(self, std_option, asi_target, rodids, overlaps, pdil, rod_pos):
        pos_node = 0.0
        k_node = self.g.kbc
        for k in range(self.g.kbc, self.g.kec):
            pos_node += self.g.hz[k]
            k_node = k
            if (pos_node - 0.01 > rod_pos):
                break

        result = SimonResult(self.g.nxya, self.g.nz)
        result.error = self.ERROR_NO
        rod_pos = pos_node

        for k in range(k_node, self.g.kbc, -1):

            self.setRodPosition(rodids, overlaps, rod_pos)
            self.calculateStatic(std_option)
            self.getResult(result)

            if rod_pos < 0.51*381:
                k_node = k
                break

            if result.asi > asi_target + self.AO_EPS:
                k_node = k
                break

            if rod_pos - self.g.hz[k] < pdil:
                result.error = self.ERROR_REACH_PDIL
                break

            rod_pos -= self.g.hz[k]

        if result.asi < asi_target - self.AO_EPS:
            result.error = self.ERROR_REACH_BOTTOM

        for i in range(len(rodids)):
            result.rod_pos[rodids[i]] = min(rod_pos + overlaps[i], self.g.core_height)

        if result.error != self.ERROR_NO:
            return result

        rod_pos += self.g.hz[k_node]

        hz_node = self.g.hz[k_node]
        nz_div = 5
        hz_div = hz_node / nz_div

        result.error = self.ERROR_REACH_BOTTOM

        for i in range(5):
            rod_pos -= hz_div
            self.setRodPosition(rodids, overlaps, rod_pos)
            self.calculateStatic(std_option)
            self.getResult(result)

            if result.asi > asi_target - self.AO_EPS:
                result.error = self.ERROR_NO
                break

        for i in range(len(rodids)):
            result.rod_pos[rodids[i]] = min(rod_pos + overlaps[i], self.g.core_height)

        return result

    def setRodPShutdown(self, power):
        position_found = 0
        for position, power_i in zip(self.shutdown_p_position, self.shutdown_p_power):
            if power >= power_i:
                position_found = position
                break
        self.setRodPosition1('P', position_found)
        return position_found

    def setRodPRestart(self, power):
        position_found = 0
        for position, power_i in zip(self.restart_p_position, self.restart_p_power):
            if power <= power_i:
                position_found = position
                break
        self.setRodPosition1('P', position_found)
        return position_found

    def getPDIL(self, power):
        if 100 >= power > 95:
            rod_position = [R5_95_65[0], R4_95_65[0], R3_42_20[0]]
        elif 95 >= power > 65:
            power_increment = 95-power
            rod_position = [R5_95_65[0]-power_increment*R5_95_65[1],
                            R4_95_65[0]-power_increment*R4_95_65[1],
                            R3_42_20[0]]
        elif 65 >= power > 50:
            power_increment = 65-power
            rod_position = [R5_65_50[0]-power_increment*R5_65_50[1],
                            R4_65_50[0]-power_increment*R4_65_50[1],
                            R3_42_20[0]]
        elif 50 >= power > 42:
            power_increment = 50-power
            rod_position = [0,
                            R4_50_42[0]-power_increment*R4_50_42[1],
                            R3_42_20[0]]
        elif 42 >= power > 20:
            power_increment = 50-power
            rod_position = [0,
                            R4_42_20[0]-power_increment*R4_42_20[1],
                            R3_42_20[0]-power_increment*R3_42_20[1]]
        elif 20 >= power >= 0:
            power_increment = 20 - power
            rod_position = [0,
                            0,
                            R3_20_0[0] - power_increment * R3_20_0[1]]
        else:
            rod_position = [381,
                            381,
                            381]
            print("Power out of range {}".format(power))
        return rod_position
