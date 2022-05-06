
from cusfam import *

class Simon:

	ASI_SEARCH_FAILED = -1.0
	ASI_SEARCH_ROD_INSERTION = -1
	ASI_SEARCH_ROD_WITHDRAWL = 1

	ERROR_NO = 0
	ERROR_REACH_PDIL = 10001
	ERROR_REACH_BOTTOM = 10002
	ERROR_REACH_TOP = 10003

	def __init__(self, file_smg, file_tset, file_rst):
		init(file_smg, file_tset, file_rst)
		self.g = SimonGeometry()
		getGeometry(self.g)

		self.g.core_height = sum(self.g.hz[self.g.kbc:self.g.kec])

	def setBurnup(self, burnup):
		setBurnup(burnup)

	def calculateStatic(self, std_option):
		calcStatic(std_option)

	def deplete(self, xe_option, sm_option, del_burn):
		deplete(xe_option, sm_option, del_burn)

	def setRodPosition1(self, rodid, position):
		setRodPosition(rodid, position)

	def setRodPosition(self, rodids, overlaps, position) :
		for i in range(len(rodids)) :
			if position + overlaps[i] < self.g.core_height + 0.001 : 
				self.setRodPosition1(rodids[i], position + overlaps[i])

	def getResult(self, result) :
		getResult(result)
	


	def searchRodPosition(self, std_option, ao_target, rodids, overlaps, pdil, rod_pos):
		crit_bak = std_option.crit

		std_option.crit = KEFF
		self.setRodPosition(rodids, overlaps, rod_pos)
		self.calculateStatic(std_option)

		result = SimonResult()
		self.getResult(result)

		if result.ao > ao_target :
			result = self.searchRodPositionDown(std_option, ao_target, rodids, overlaps, pdil, rod_pos)
		else :
			result = self.searchRodPositionUp(std_option, ao_target, rodids, overlaps, pdil, rod_pos)

		std_option.crit = CBC
		self.calculateStatic(std_option)
		self.getResult(result)

		std_option.crit = crit_bak

		return result



	def searchRodPositionUp(self, std_option, ao_target, rodids, overlaps, pdil, rod_pos) :

		pos_node = 0.0
		k_node = self.g.kbc
		for k in range(self.g.kbc, self.g.kec) :
			pos_node += self.g.hz[k]

			if(pos_node - 0.01 > rod_pos) : 
				k_node = k
				break

		result = SimonResult()

		k_node-=1
		pos_node -= self.g.hz[k_node]
		rod_pos = pos_node
		
		for k in range(k_node,self.g.kec) :
			rod_pos += self.g.hz[k]

			self.setRodPosition(rodids, overlaps, rod_pos)
			self.calculateStatic(std_option)
			self.getResult(result)

			if result.ao > ao_target - 0.001 :
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

			if result.ao > ao_target - 0.001 : 
				break

		if abs(result.ao - ao_target) <= 0.001 : 
			result.error = self.ERROR_NO
		else :
			result.error = self.ERROR_REACH_TOP
	
		for i in range(len(rodids)) :
			result.rod_pos[rodids[i]] = min(rod_pos + overlaps[i],self.g.core_height)

		return result


	def searchRodPositionDown(self, std_option, ao_target, rodids, overlaps, pdil, rod_pos) :
		pos_node = 0.0
		k_node = self.g.kbc
		for k in range(self.g.kbc, self.g.kec) :
			pos_node += self.g.hz[k]
			k_node = k
			if(pos_node - 0.01 > rod_pos) : 
				break

		result = SimonResult()
		rod_pos = pos_node
		
		for k in range(k_node,self.g.kbc,-1) :
			self.setRodPosition(rodids, overlaps, rod_pos)
			self.calculateStatic(std_option)
			self.getResult(result)

			if result.ao < ao_target + 0.001 :
				k_node = k
				break

			if rod_pos < pdil : 
				result.error = self.ERROR_REACH_PDIL
				return result

			rod_pos -= self.g.hz[k]


		rod_pos += self.g.hz[k_node]

		hz_node = self.g.hz[k_node]
		nz_div = 5
		hz_div = hz_node / nz_div
		
		for i in range(5) :
			rod_pos -= hz_div
			self.setRodPosition(rodids, overlaps, rod_pos)
			self.calculateStatic(std_option)
			self.getResult(result)

			if result.ao < ao_target + 0.001 : 
				break

		if abs(result.ao - ao_target) <= 0.001 : 
			result.error = self.ERROR_NO
		else :
			result.error = self.ERROR_REACH_TOP
	
		for i in range(len(rodids)) :
			result.rod_pos[rodids[i]] = min(rod_pos + overlaps[i],self.g.core_height)

		return result


	def setDefinedRodPosition(self, rodids, position) :
		for i in range(len(rodids)) :
			self.setRodPosition1(rodids[i], position[i]*self.g.core_height*0.01)

	def searchASI(self, std_option, rodids, rod_pos) :
		crit_bak = std_option.crit

		std_option.crit = KEFF
		self.calculateStatic(std_option)

		result = SimonResult()

		for i in range(len(rodids)):
			result.rod_pos[rodids[i]] = rod_pos[i] * self.g.core_height / 100.0

		self.setDefinedRodPosition(rodids, rod_pos)
		std_option.crit = CBC
		self.calculateStatic(std_option)
		self.getResult(result)
		
		return result