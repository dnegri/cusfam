from cusfam import *

class Simon:

	ASI_SEARCH_FAILED = -1.0
	ASI_SEARCH_ROD_INSERTION = -1
	ASI_SEARCH_ROD_WITHDRAWL = 1

	ERROR_NO = 0
	ERROR_REACH_PDIL = 10001
	ERROR_REACH_BOTTOM = 10002
	ERROR_REACH_TOP = 10003
	ERROR_REACH_BOTTOM = 10004

	AO_EPS = 0.002


	def __init__(self, file_smg, file_tset, file_ff, file_rst):
		self.s = init(file_smg, file_tset, file_ff)
		self.g = SimonGeometry()
		getGeometry(self.s, self.g)
		self.file_rst = file_rst

	def setBurnupPoints(self, burnups):
		setBurnupPoints(self.s, burnups)

	def setBurnup(self, burnup):
		setBurnup(self.s, self.file_rst, burnup)

	def calculateStatic(self, std_option):
		calcStatic(self.s, std_option)

	def calculatePinPower(self):
		calcPinPower(self.s)

	def deplete(self, xe_option, sm_option, del_burn):
		deplete(self.s, xe_option, sm_option, del_burn)

	def depleteByTime(self, xe_option, sm_option, tsec):
		depleteByTime(self.s, xe_option, sm_option, tsec)

	def depleteXeSm(self, xe_option, sm_option, tsec):
		depleteXeSm(self.s, xe_option, sm_option, tsec)

	def setRodPosition1(self, rodid, position):
		setRodPosition(self.s, rodid, position)

	def setRodPosition(self, rodids, overlaps, position) :
		for i in range(len(rodids)) :
			pos = min(position + overlaps[i], self.g.core_height)
			self.setRodPosition1(rodids[i], pos)

	def getResult(self, result) :
		getResult(self.s, result)
	
	def getShutDownMargin(self, std_option, rodids, overlaps, pdil, rod_pos, stuck_rods) :
		crit_bak = std_option.crit
		xe_bak = std_option.xenon

		std_option.crit = KEFF
		std_option.xenon = XE_TR

		self.setRodPosition(rodids, overlaps, rod_pos)
		self.calculateStatic(std_option)
		result = SimonResult(self.g.nxya, self.g.nz)
		self.getResult(result)
		out_eigv = result.eigv

		self.setRodPosition(['R','B','A','P'], [0] * 4, 0.0)
		for stuck_rod in stuck_rods :
			self.setRodPosition1(stuck_rod, self.g.core_height)

		self.calculateStatic(std_option)
		self.getResult(result)
		in_eigv = result.eigv

		std_option.crit = crit_bak
		std_option.xenon = xe_bak 

		sdm = out_eigv-in_eigv

		return sdm


	def searchRodPosition(self, std_option, asi_target, rodids, overlaps, pdil, rod_pos):
		crit_bak = std_option.crit

		std_option.crit = KEFF
		self.setRodPosition(rodids, overlaps, rod_pos)
		self.calculateStatic(std_option)

		result = SimonResult(self.g.nxya, self.g.nz)
		self.getResult(result)

		if result.asi < asi_target :
			result = self.searchRodPositionDown(std_option, asi_target, rodids, overlaps, pdil, rod_pos)
		else :
			result = self.searchRodPositionUp(std_option, asi_target, rodids, overlaps, pdil, rod_pos)

		std_option.crit = CBC
		self.calculateStatic(std_option)
		self.getResult(result)

		std_option.crit = crit_bak

		return result



	def searchRodPositionUp(self, std_option, asi_target, rodids, overlaps, pdil, rod_pos) :

		pos_node = 0.0
		k_node = self.g.kbc
		for k in range(self.g.kbc, self.g.kec) :
			pos_node += self.g.hz[k]
			k_node = k
			if(pos_node - 0.01 > rod_pos) : 
				break

		result = SimonResult(self.g.nxya, self.g.nz)
		result.error = self.ERROR_NO

		pos_node -= self.g.hz[k_node]
		rod_pos = pos_node
		
		for k in range(k_node,self.g.kec) :
			rod_pos += self.g.hz[k]

			self.setRodPosition(rodids, overlaps, rod_pos)
			self.calculateStatic(std_option)
			self.getResult(result)

			k_node = k
			if result.asi < asi_target - self.AO_EPS :
				break

			if rod_pos < pdil : 
				result.error = self.ERROR_REACH_PDIL
				break

		if result.asi > asi_target - self.AO_EPS:
			result.error = self.ERROR_REACH_TOP

		for i in range(len(rodids)) :
			result.rod_pos[rodids[i]] = min(rod_pos + overlaps[i],self.g.core_height)

		if result.error != self.ERROR_NO : 
			return result

		rod_pos -= self.g.hz[k_node]

		hz_node = self.g.hz[k_node]
		nz_div = 5
		hz_div = hz_node / nz_div
		
		result.error = self.ERROR_REACH_TOP

		for i in range(5) :
			rod_pos += hz_div
			self.setRodPosition(rodids, overlaps, rod_pos)
			self.calculateStatic(std_option)
			self.getResult(result)

			if result.asi < asi_target + self.AO_EPS : 
				result.error = self.ERROR_NO
				break

		for i in range(len(rodids)) :
			result.rod_pos[rodids[i]] = min(rod_pos + overlaps[i],self.g.core_height)

		return result


	def searchRodPositionDown(self, std_option, asi_target, rodids, overlaps, pdil, rod_pos) :
		pos_node = 0.0
		k_node = self.g.kbc
		for k in range(self.g.kbc, self.g.kec) :
			pos_node += self.g.hz[k]
			k_node = k
			if(pos_node - 0.01 > rod_pos) : 
				break

		result = SimonResult(self.g.nxya, self.g.nz)
		result.error = self.ERROR_NO
		rod_pos = pos_node
		
		for k in range(k_node,self.g.kbc,-1) :
			self.setRodPosition(rodids, overlaps, rod_pos)
			self.calculateStatic(std_option)
			self.getResult(result)

			if result.asi > asi_target + self.AO_EPS :
				k_node = k
				break

			if rod_pos < pdil : 
				result.error = self.ERROR_REACH_PDIL
				break

			rod_pos -= self.g.hz[k]

		if result.asi < asi_target - self.AO_EPS:
			result.error = self.ERROR_REACH_BOTTOM

		for i in range(len(rodids)) :
			result.rod_pos[rodids[i]] = min(rod_pos + overlaps[i],self.g.core_height)

		if result.error != self.ERROR_NO : 
			return result

		rod_pos += self.g.hz[k_node]

		hz_node = self.g.hz[k_node]
		nz_div = 5
		hz_div = hz_node / nz_div
		
		result.error = self.ERROR_REACH_BOTTOM

		for i in range(5) :
			rod_pos -= hz_div
			self.setRodPosition(rodids, overlaps, rod_pos)
			self.calculateStatic(std_option)
			self.getResult(result)

			if result.asi > asi_target - self.AO_EPS : 
				result.error = self.ERROR_NO
				break

		for i in range(len(rodids)) :
			result.rod_pos[rodids[i]] = min(rod_pos + overlaps[i],self.g.core_height)

		return result