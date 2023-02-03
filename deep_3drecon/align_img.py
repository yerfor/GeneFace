import numpy as np
import cv2
from scipy.io import loadmat,savemat
from PIL import Image

#TODO: refractor, support get translation from lm 5 and translate lm 68, put this file outof deep3drecon
#calculating least square problem
def POS(xp,x):
	npts = xp.shape[1]

	A = np.zeros([2*npts,8])

	A[0:2*npts-1:2,0:3] = x.transpose()
	A[0:2*npts-1:2,3] = 1

	A[1:2*npts:2,4:7] = x.transpose()
	A[1:2*npts:2,7] = 1;

	b = np.reshape(xp.transpose(),[2*npts,1])

	k,_,_,_ = np.linalg.lstsq(A,b)

	R1 = k[0:3]
	R2 = k[4:7]
	sTx = k[3]
	sTy = k[7]
	s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2
	t = np.stack([sTx,sTy],axis = 0)

	return t,s

def process_img(img,lm,t,s,target_size = 224.):
	w0,h0 = img.size
	w = (w0/s*102).astype(np.int32)
	h = (h0/s*102).astype(np.int32)
	img = img.resize((w,h),resample = Image.BICUBIC)

	left = (w/2 - target_size/2 + float((t[0] - w0/2)*102/s)).astype(np.int32)
	right = left + target_size
	up = (h/2 - target_size/2 + float((h0/2 - t[1])*102/s)).astype(np.int32)
	below = up + target_size

	img = img.crop((left,up,right,below))
	img = np.array(img)
	img = img[:,:,::-1] #RGBtoBGR
	img = np.expand_dims(img,0)
	lm = np.stack([lm[:,0] - t[0] + w0/2,lm[:,1] - t[1] + h0/2],axis = 1)/s*102
	lm = lm - np.reshape(np.array([(w/2 - target_size/2),(h/2-target_size/2)]),[1,2])

	return img,lm


def align(img, lm, lm3D):
	'''
		Feed in Opencv Image
        given image, lm, and standard lm, align the image
	'''

	img = img[:,:,::-1]
	img = Image.fromarray(img)
	w0,h0 = img.size

	lm = np.stack([lm[:,0],h0 - 1 - lm[:,1]], axis = 1)

	# calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
	t,s = POS(lm.transpose(),lm3D.transpose())

	# processing the image
	img_new,lm_new = process_img(img,lm,t,s)
	lm_new = np.stack([lm_new[:,0],223 - lm_new[:,1]], axis = 1)
	trans_params = np.array([w0,h0,102.0/s,t[0],t[1]])

	return img_new,lm_new,trans_params

def align_video(img_array, lm_array, lm3D):
	t_list = []
	s_list = []
	w0,h0 = Image.fromarray(img_array[0]).size
	for i in range(len(lm_array)):
		lm = lm_array[i]
		lm = np.stack([lm[:,0],h0 - 1 - lm[:,1]], axis = 1)
		t,s = POS(lm.transpose(),lm3D.transpose())
		t_list.append(t)
		s_list.append(s)

	s = np.mean(np.array(s_list))

	img_list = []
	lm_new_list = []
	trans_list = []
	for i in range(len(img_array)):
		t = t_list[i]
		img = img_array[i][:,:,::-1]
		img = Image.fromarray(img)
		img_new, lm_new = process_img(img, lm_array[i], t_list[i], s)
		lm_new = np.stack([lm_new[:,0],223 - lm_new[:,1]], axis = 1)
		trans_params = np.array([w0,h0,102.0/s,t[0],t[1]])
		img_list.append(img_new[0])
		lm_new_list.append(lm_new)
		trans_list.append(trans_params)

	return np.array(img_list), np.array(lm_new_list), np.array(trans_list)


def align_lm68(lm5_list, lm68_list, lm3D, w0, h0, target_size = 224.):
	'''
		batch wise lm and lm 68
	'''
	lm68_new_list = []
	for i in range(len(lm5_list)):
		lm5 = lm5_list[i]
		lm68 = lm68_list[i]
		lm5 = np.stack([lm5[:,0],h0 - 1 - lm5[:,1]], axis = 1)
		lm68 = np.stack([lm68[:,0],h0 - 1 - lm68[:,1]], axis = 1)
		t,s = POS(lm5.transpose(),lm3D.transpose())

		w = (w0/s*102).astype(np.int32)
		h = (h0/s*102).astype(np.int32)
		lm68_new = np.stack([lm68[:,0] - t[0] + w0/2,lm68[:,1] - t[1] + h0/2],axis = 1)/s*102
		lm68_new = lm68_new - np.reshape(np.array([(w/2 - target_size/2),(h/2-target_size/2)]),[1,2])
		lm68_new = np.stack([lm68_new[:,0], 223 - lm68_new[:,1]], axis = 1)
		lm68_new_list.append(lm68_new)

	return np.array(lm68_new_list)