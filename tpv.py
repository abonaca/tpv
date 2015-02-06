import numpy as np

def transform(px, py, header):
	"""Transform from pixel space to ra,dec assuming TAN (gnomonic) or TPV projection
	Parameters:
	px - pixel x coordinate
	py - pixel y coordinate
	header - image header, needs to contain fields: crpix1, crpix2, cd1_1. cd2_2, crval1, crval2, ctype1
	Returns: (ra, dec)"""
	
	rx = header['crpix1']
	ry = header['crpix2']
	cd11 = header['cd1_1']
	if('cd1_2' in header):
		cd12 = header['cd1_2']
	else:
		cd12 = 0.
	if('cd2_1' in header):
		cd21 = header['cd2_1']
	else:
		cd21 = 0.
	cd22 = header['cd2_2']
	ar = header['crval1']
	dr = header['crval2']
	
	x = cd11*(px-rx) + cd12*(py-ry)
	y = cd21*(px-rx) + cd22*(py-ry)
	
	# apply field distortion correction
	ct = header['ctype1']
	if(ct.find('TPV')!=-1):
		x, y = distortion(x, y, header)
	
	# make sure angles between 0, 360 deg
	phi = np.arctan(-x/y)
	ind = ((x>0) & (y>0)) | ((x<0) & (y>0))
	phi[ind] = phi[ind] + np.pi
	ind = ((x<0) & (y<0))
	phi[ind] = phi[ind] + 2*np.pi
	
	rt = np.radians(np.sqrt(x**2 + y**2))
	theta = np.arctan(1./rt)
	
	sp = np.sin(phi)
	cp = np.cos(phi)
	st = np.sin(theta)
	ct = np.cos(theta)
	sd = np.sin(np.radians(dr))
	cd = np.cos(np.radians(dr))
	
	a = ar + np.degrees( np.arctan( ct*sp / ( st*cd + ct*cp*sd ) ) )
	d = np.degrees( np.arcsin( st*sd - ct*cp*cd ) )
	
	return (a, d)
	
def distortion(xi, eta, head):
	"""Correct projection plane coordinates for field distortion"""
	
	# Distortion coefficients (default to 'TAN' projection)
	Nc = 40
	pv1 = np.zeros(Nc)
	pv2 = np.zeros(Nc)
	pv1[1] = pv2[1] = 1
	
	# Update coefficients present in the header (for 'TPV' projection)
	for i in range(Nc):
		name1 = "pv1_%d"%i
		name2 = "pv2_%d"%i
		if name1 in head:
			pv1[i] = head[name1]
		if name2 in head:
			pv2[i] = head[name2]
	
	# Apply correction (source: http://iraf.noao.edu/projects/ccdmosaic/tpv.html)
	r = np.sqrt(xi**2 + eta**2)
	
	xi = (pv1[0] + pv1[1] * xi + pv1[2] * eta + pv1[3] * r + 
	pv1[4]* xi**2 + pv1[5] * xi * eta + pv1[6] * eta**2 + 
	pv1[7] * xi**3 + pv1[8] * xi**2 * eta + pv1[9] * xi * eta**2 + pv1[10] * eta**3 + pv1[11] * r**3 + 
	pv1[12] * xi**4 + pv1[13] * xi**3 * eta + pv1[14] * xi**2 * eta**2 + pv1[15] * xi * eta**3 + pv1[16] * eta**4 + 
	pv1[17] * xi**5 + pv1[18] * xi**4 * eta + pv1[19] * xi**3 * eta**2 + 
	pv1[20] * xi**2 * eta**3 + pv1[21] * xi * eta**4 + pv1[22] * eta**5 + pv1[23] * r**5 + 
	pv1[24] * xi**6 + pv1[25] * xi**5 * eta + pv1[26] * xi**4 * eta**2 + pv1[27] * xi**3 * eta**3 + 
	pv1[28] * xi**2 * eta**4 + pv1[29] * xi * eta**5 + pv1[30] * eta**6 + 
	pv1[31] * xi**7 + pv1[32] * xi**6 * eta + pv1[33] * xi**5 * eta**2 + pv1[34] * xi**4 * eta**3 + 
	pv1[35] * xi**3 * eta**4 + pv1[36] * xi**2 * eta**5 + pv1[37] * xi * eta**6 + pv1[38] * eta**7 + pv1[39] * r**7)

	eta = (pv2[0] + pv2[1] * eta + pv2[2] * xi + pv2[3] * r + 
	pv2[4] * eta**2 + pv2[5] * eta * xi + pv2[6] * xi**2 + 
	pv2[7] * eta**3 + pv2[8] * eta**2 * xi + pv2[9] * eta * xi**2 + pv2[10] * xi**3 + pv2[11] * r**3 + 
	pv2[12] * eta**4 + pv2[13] * eta**3 * xi + pv2[14] * eta**2 * xi**2 + pv2[15] * eta * xi**3 + pv2[16] * xi**4 + 
	pv2[17] * eta**5 + pv2[18] * eta**4 * xi + pv2[19] * eta**3 * xi**2 + 
	pv2[20] * eta**2 * xi**3 + pv2[21] * eta * xi**4 + pv2[22] * xi**5 + pv2[23] * r**5 + 
	pv2[24] * eta**6 + pv2[25] * eta**5 * xi + pv2[26] * eta**4 * xi**2 + pv2[27] * eta**3 * xi**3 + 
	pv2[28] * eta**2 * xi**4 + pv2[29] * eta * xi**5 + pv2[30] * xi**6 + 
	pv2[31] * eta**7 + pv2[32] * eta**6 * xi + pv2[33] * eta**5 * xi**2 + pv2[34] * eta**4 * xi**3 + 
	pv2[35] * eta**3 * xi**4 + pv2[36] * eta**2 * xi**5 + pv2[37] * eta * xi**6 + pv2[38] * xi**7 + pv2[39] * r**7)

	return (xi, eta)
