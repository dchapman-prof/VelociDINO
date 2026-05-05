legendre_scales = (2.0, 0.6666666666666666, 0.4, 0.2857142857142858, 
						0.2222222222222223, 0.1818181818181799, 
						0.1538461538461604, 0.1333333333333635, 
						 0.1176470588234224, 0.1052631578950241, 
						 0.1081811814074172)

#---------						 
# Calculate Legendre Polynomials
#---------						 
NP = 11
SX = 64
P = torch.zeros((NP,SX), dtype=torch.float32, 
device=device, requires_grad=False)
P[0] = 1.0
P[1] = torch.arange(SX, dtype=torch.float32, 
device=device, requires_grad=False)
P[1] = (P[1] + 0.5) / SX
P[1] = 2*P[1] - 1.0

for p in range(2,NP):
	P[p] = ( (2.0 * (p-1))*P[1]*P[p-1] - (p-1)*P[p-2] ) / (p)
	
#---------						 
# Normalize the Legendre polynomials
#---------						 

legendre_scales = torch.tensor(legendre_scales, device=device, requires_grad=False, dtype=torch.float32)
legendre_scales = torch.reshape(legendre_scales, (NP, 1))
P = P / torch.sqrt(legendre_scales)

#---------						 

#---------						 


