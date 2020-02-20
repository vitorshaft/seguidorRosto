import RPi.GPIO as GPIO
import time

class andar:
	def __init__(self):
		pass
	
	def frente(self,passos):
		GPIO.setmode(GPIO.BOARD)
		control_pins = [7,11,13,15]
		control_esq = [31,33,35,37]
		for pin in control_pins:
			GPIO.setup(pin, GPIO.OUT)
			GPIO.output(pin, 0)
		for p in control_esq:
			GPIO.setup(p, GPIO.OUT)
			GPIO.output(p, 0)
		'''
		halfstep_seq = [
			[1,0,0,0],
			[1,1,0,0],
			[0,1,0,0],
			[0,1,1,0],
			[0,0,1,0],
			[0,0,1,1],
			[0,0,0,1],
			[1,0,0,1]
		]
		
		singlestep_seq = [
			[1,0,0,0],
			[0,1,0,0],
			[0,0,1,0],
			[0,0,0,1]
		]
		'''
		fullstep_seq = [
			[1,1,0,0],
			[0,1,1,0],
			[0,0,1,1],
			[1,0,0,1]
		]
		
		for i in range(passos):
			#for halfstep in range(8):	#maior precisao, menor velocidade
			for singlestep in range(4):	#maior velocidade, menor precisao
				'''
				for pin in range(4):
					GPIO.output(control_pins[pin], halfstep_seq[halfstep][pin])
					GPIO.output(control_esq[pin], halfstep_seq[halfstep][pin])
					time.sleep(0.001)
				
				for pin in range(4):
					GPIO.output(control_pins[pin], singlestep_seq[singlestep][pin])
					GPIO.output(control_esq[pin], singlestep_seq[singlestep][pin])
					time.sleep(0.001)
				'''
				for pin in range(4):
					GPIO.output(control_pins[pin], fullstep_seq[singlestep][pin])
					GPIO.output(control_esq[pin], fullstep_seq[singlestep][pin])
					time.sleep(0.001)
		#frente(512)
		GPIO.cleanup()
###############################################################################
	def esqRad(self,passos):
		GPIO.setmode(GPIO.BOARD)
		control_esq = [12,16,18,22]
		for p in control_esq:
			GPIO.setup(p, GPIO.OUT)
			GPIO.output(p, 0)
		fullstep_seq = [
			[1,1,0,0],
			[0,1,1,0],
			[0,0,1,1],
			[1,0,0,1]
		]
		for i in range(passos):
			#for halfstep in range(8):	#maior precisao, menor velocidade
			for singlestep in range(4):	#maior velocidade, menor precisao
				for pin in range(4):
					GPIO.output(control_esq[pin], fullstep_seq[singlestep][pin])
					time.sleep(0.0005)
					
		#frente(512)
		GPIO.cleanup()
#################################################################################
	def esqAx(self,graus):
		GPIO.setmode(GPIO.BOARD)
		control_pins = [7,11,13,15]
		control_esq = [31,33,35,37]
		for pin in control_pins:
			GPIO.setup(pin, GPIO.OUT)
			GPIO.output(pin, 0)
		for p in control_esq:
			GPIO.setup(p, GPIO.OUT)
			GPIO.output(p, 0)
		
		fullstep_seq = [
			[1,1,0,0],
			[0,1,1,0],
			[0,0,1,1],
			[1,0,0,1]
		]
		passos = 2.667*graus
		for i in range(int(passos)):
			#for halfstep in range(8):	#maior precisao, menor velocidade
			for singlestep in range(4):	#maior velocidade, menor precisao
				
				for pin in range(4):
					GPIO.output(control_pins[pin], fullstep_seq[singlestep*(-1)][pin])
					GPIO.output(control_esq[pin], fullstep_seq[singlestep][pin])
					time.sleep(0.001)
		#frente(512)
		GPIO.cleanup()
#################################################################################
	def dirRad(self,passos):
		GPIO.setmode(GPIO.BOARD)
		control_pins = [7,11,13,15]
		for p in control_pins:
			GPIO.setup(p, GPIO.OUT)
			GPIO.output(p, 0)
		fullstep_seq = [
			[1,1,0,0],
			[0,1,1,0],
			[0,0,1,1],
			[1,0,0,1]
		]
		for i in range(passos):
			#for halfstep in range(8):	#maior precisao, menor velocidade
			for singlestep in range(4):	#maior velocidade, menor precisao
				for pin in range(4):
					GPIO.output(control_pins[pin], fullstep_seq[singlestep][pin])
					time.sleep(0.001)
					
		#frente(512)
		GPIO.cleanup()
####################################################################################
	def dirAx(self,graus):
		GPIO.setmode(GPIO.BOARD)
		control_pins = [7,11,13,15]
		control_esq = [31,33,35,37]
		for pin in control_pins:
			GPIO.setup(pin, GPIO.OUT)
			GPIO.output(pin, 0)
		for p in control_esq:
			GPIO.setup(p, GPIO.OUT)
			GPIO.output(p, 0)
		
		fullstep_seq = [
			[1,1,0,0],
			[0,1,1,0],
			[0,0,1,1],
			[1,0,0,1]
		]
		passos = 2.667*graus
		for i in range(int(passos)):
			#for halfstep in range(8):	#maior precisao, menor velocidade
			for singlestep in range(4):	#maior velocidade, menor precisao
				
				for pin in range(4):
					GPIO.output(control_pins[pin], fullstep_seq[singlestep][pin])
					GPIO.output(control_esq[pin], fullstep_seq[singlestep*(-1)][pin])
					time.sleep(0.001)
		#frente(512)
		GPIO.cleanup()
#####################################################################################
	def tras(self,passos):
		GPIO.setmode(GPIO.BOARD)
		control_pins = [7,11,13,15]
		control_esq = [31,33,35,37]
		for pin in control_pins:
			GPIO.setup(pin, GPIO.OUT)
			GPIO.output(pin, 0)
		for p in control_esq:
			GPIO.setup(p, GPIO.OUT)
			GPIO.output(p, 0)
		
		fullstep_seq = [
			[1,1,0,0],
			[0,1,1,0],
			[0,0,1,1],
			[1,0,0,1]
		]
		
		for i in range(passos):
			#for halfstep in range(8):	#maior precisao, menor velocidade
			for singlestep in range(4):	#maior velocidade, menor precisao
				
				for pin in range(4):
					GPIO.output(control_pins[pin], fullstep_seq[singlestep*(-1)][pin])
					GPIO.output(control_esq[pin], fullstep_seq[singlestep*(-1)][pin])
					time.sleep(0.001)
		#frente(512)
		GPIO.cleanup()
