#!/usr/bin/env python
# coding: utf-8

# ## Cell Center Detection using Phathom (Parallel)
# - This notebook is to analyze cell center detection algorithm in Phathom for Microglia/Nuclei dataset
# - run detection in parallel with Zarr stack

# ### 0. Imports

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import platform
print(platform.python_version())
import sys
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
from collections import OrderedDict
from datetime import date

# PHATHOM
from phathom import io
from phathom.segmentation.segmentation import find_centroids
from phathom.phenotype.celltype import nucleus_probability, nuclei_centers_probability, calculate_eigvals
from phathom.phenotype.celltype import detect_nuclei_parallel
from phathom.utils import extract_box
# BM
from analysis.cellCentersAnalyzer import CellCenterAnalyzer as CCA
from utils.train.preprocessor import TrainPreprocessor
from ccd.cell_center_detection import BM_CCD
from utils.params import DataGenParams
from utils.data.data_generator import DataGenerator
from utils.data import preprocessing
from utils.const import NormalizationType, StainChannel

bmp = preprocessing.BMPreprocessing()


# ### 1. Global Params

# In[2]:


workdir = '/data2/raw/microglia_proj/361_CKp25_GENUS_10x'
datatype = StainChannel.IBA1
paramf = os.path.join(workdir, 'params.json')
with open(paramf) as fp:
    params = json.load(fp)
params


# In[3]:


params_byType = params[datatype]
params_byType


# In[4]:


zarr_path = os.path.join(workdir, params_byType['rel_path'], params['zarr_rel_path'])
zarr = io.zarr.open(zarr_path)
zarr.shape, zarr.chunks


# In[5]:


#norm_percentile = (0.5, 99.5)
norm_percentile = (0.05, 99.5)

# for CCD on segmented mask maps
#norm_percentile = None
G_SLICE_NO = 20


# ### 2. CCD on subvolume (non-parallel)

# In[6]:


ccd_chn = BM_CCD(paramf, datatype, G_SLICE_NO)
#ccd_chn2 = BM_CCD(paramf, StainChannel.IBA1_SEG, G_SLICE_NO)
if False: #params[StainChannel.AUTOFLUORESCENCE]['rel_path'] == '':
    ccd_af = None
else:
    ccd_af = BM_CCD(paramf, StainChannel.AUTOFLUORESCENCE, G_SLICE_NO)


# #### Preprocess and show a slice

# In[7]:


ccd_chn.preprocess(normByPercentile=norm_percentile, clip=True)
if ccd_af is not None:
    ccd_af.preprocess(normByPercentile=norm_percentile, clip=True)


# In[8]:


chn_slice = ccd_chn.get_a_slice(G_SLICE_NO, raw=True)
chn_slice2 = None
if ccd_af:
    af_slice = ccd_af.get_a_slice(G_SLICE_NO, raw=True)
print("chn_slice.shape: ", chn_slice.shape)
    
fig, ax = plt.subplots(figsize=(20, 8), nrows=1, ncols=3, sharex=False)
ax[0].imshow(chn_slice, cmap='viridis')
ax[0].set_title('%s-raw'%datatype)
if chn_slice2 is not None:
    ax[1].imshow(chn_slice2)
    ax[1].set_title('2ndChn-raw')
elif ccd_af:
    ax[1].imshow(af_slice)
    ax[1].set_title('AF-raw')
    ax[2].imshow(chn_slice-af_slice)
    ax[2].set_title('subtracted-raw')
plt.show()

chn_slice = ccd_chn.get_a_slice(G_SLICE_NO)
if ccd_af:
    af_slice = ccd_af.get_a_slice(G_SLICE_NO)
fig, ax = plt.subplots(figsize=(20, 8), nrows=1, ncols=3, sharex=False)
ax[0].imshow(chn_slice, cmap='viridis')
ax[0].set_title('%s-processed'%datatype)
if chn_slice2 is not None:
    ax[1].imshow(chn_slice2)
    ax[1].set_title('2ndChn-raw')
elif ccd_af:
    ax[1].imshow(af_slice)
    ax[1].set_title('AF-processed')
    ax[2].imshow(chn_slice-af_slice)
    ax[2].set_title('subtracted')
plt.show()


# In[9]:


def print_stats(ccd):
    print("[raw]: mean: {}, std: {}, min: {}, max: {}".format(np.mean(ccd.vol_raw),                                                        np.std(ccd.vol_raw),                                                        np.min(ccd.vol_raw),                                                        np.max(ccd.vol_raw)))
    print("[norm]: mean: {}, std: {}, min: {}, max: {}".format(np.mean(ccd.vol),                                                        np.std(ccd.vol),                                                        np.min(ccd.vol),                                                        np.max(ccd.vol)))


# In[10]:


print_stats(ccd_chn)
if ccd_af:
    print_stats(ccd_af)


# In[11]:


slice_no = ccd_chn.slice_no
if ccd_af:
    subtracted = ccd_chn.vol - ccd_af.vol
    subtracted_processed = ccd_chn.tPrep.preprocess_all(subtracted.copy())
    fig, ax = plt.subplots(figsize=(12, 8), nrows=1, ncols=2)
    ax[0].imshow(subtracted[slice_no], cmap='viridis')
    ax[0].set_title('subtracted')
    ax[1].imshow(subtracted_processed[slice_no])
    ax[1].set_title('subtracted-processed')
    plt.show()
    print(np.mean(subtracted_processed))


# In[12]:


pre_normalized = True
#pre_normalized = False

if pre_normalized:
    if False:
        sigma = 9.0
        steepness = 1500
        offset = 0.0001
        threshold = 0.7
        min_dist = 14
    else:
        # for normalized data
        sigma = 4.5
        steepness = 1300
        offset = 0.0001
        #threshold = 0.3
        threshold = 0.8
        min_dist = 3
        #mean = np.mean(subtracted_processed)
        #stdev = np.std(subtracted_processed)
        #mean = np.mean(ccd_gfp.vol)
        #stdev = np.std(ccd_gfp.vol)
else:
    # for raw 16-bit data
    sigma = (2.5, 4.0, 4.0)
    steepness = 5000
    offset = 0.0001
    threshold = 0.2
    min_dist = 3
    mean = params_byType['mean']
    stdev = params_byType['std']
    
#sigma = (1.8, 3.0, 3.0)
#steepness = 5000
#offset = -0.0001
#I0 = 500
#stdev = 1e-5


# In[13]:


#ccd_chn.detect_centers(sigma, steepness, offset, threshold, min_dist, mean, stdev)
ccd_chn.detect_centers(sigma, steepness, offset, threshold, min_dist, None, None)
len(ccd_chn.centroids)


# In[14]:


len(ccd_chn.centroids), len(ccd_chn.cs)
new_cs = BM_CCD.get_centroids_in_range(ccd_chn.centroids, ccd_chn.slice_no, 4)
len(new_cs)
ccd_chn.viz_all(ccd_chn.get_a_slice(), new_cs, None, [], reverse=True, s=5)


# #### Detect Center again with Subtraction

# In[15]:


if ccd_af:
    ccd_chn.detect_centers(sigma, steepness, offset, threshold, min_dist, arr_to_subtract=ccd_af.vol)
    len(ccd_chn.centroids)


# In[16]:


if ccd_af:
    ccd_chn.slice_no = 20
    len(ccd_chn.centroids), len(ccd_chn.cs)
    new_cs = BM_CCD.get_centroids_in_range(ccd_chn.centroids, ccd_chn.slice_no, 4)
    len(new_cs)
    ccd_chn.viz_all(ccd_chn.get_a_slice(), new_cs, None, [], reverse=True, s=5)


# ### 3. CCD on whole volume (parallel)

# In[17]:


if ccd_af:
    af_zarr_path = os.path.join(workdir, params['AF']['rel_path'], params['zarr_rel_path'])
    af_zarr = io.zarr.open(af_zarr_path)
    print(af_zarr.shape, af_zarr.chunks)
else:
    af_zarr = None


# In[18]:


from scipy import stats

prob_map = io.zarr.new_zarr(os.path.join(workdir, params_byType['rel_path'], 'prob_map.zarr'), 
                            zarr.shape, zarr.chunks, dtype=np.float32)

subzarr = zarr[params['zr'][0]:params['zr'][1], params['yr'][0]:params['yr'][1], 
               params['xr'][0]:params['xr'][1]]
print('subzarr shape: ', subzarr.shape)
s1 = np.asarray(subzarr[1], dtype=np.float32)
print('sub GFP: ', s1.shape, s1.max(), s1.min(), s1.dtype)

if ccd_af:
    af_subzarr = af_zarr[params['zr'][0]:params['zr'][1], params['yr'][0]:params['yr'][1], 
                         params['xr'][0]:params['xr'][1]]
    s2 = np.asarray(af_subzarr[1], dtype=np.float32)
    print('sub2 AF: ', s2.shape, s2.max(), s2.min(), s2.dtype)
    
else:
    s2 = None

m1 = stats.mode(s1, axis=None)
if s2 is not None:
    m2 = stats.mode(s2, axis=None)
else:
    m2 = Non3
print("mod: ", m1, m2)

p1 = np.percentile(s1, 0.5)
p2 = np.percentile(s2, 0.5) if s2 is not None else None
print("percentile 0.5%: ", p1, p2)

n1 = np.percentile(s1, 99.5)
n2 = np.percentile(s2, 99.5) if s2 is not None else None
print("percentile 99.5%: ", n1, n2)


fig, ax = plt.subplots(figsize=(12, 8), nrows=2, ncols=2)
ax[0][0].imshow(s1, clim=[p1, n1])
ax[0][0].set_title('GFP zarr')
ax[0][1].hist(s1.ravel(), bins=256)

if s2 is not None:
    ax[1][0].imshow(s2, clim=[p1, n2])
    ax[1][0].set_title('AF zarr')
    ax[1][1].hist(s2.ravel(), bins=256)
plt.show()


# In[19]:


sigma, steepness, offset, threshold, min_dist


# In[20]:


mean = None
stdev = None
clip = True
mean, stdev, clip


# In[21]:


#prob_thresh = 0.5
prob_thresh = threshold
#min_dist = 3
min_intensity = 100
#min_intensity = 0
chunk_size = zarr.chunks
overlap = 16

prob_thresh, min_dist, min_intensity, chunk_size, overlap


# In[22]:


normalize = True
norm_percentile = None
#norm_min = params['min']
#norm_max = params['max']
norm_min = 200
norm_max = 3000
#norm_min = None
#norm_max = None

# ASSERT! Either norm_percentile or (norm_min, norm_max) should be None!
normalize, norm_percentile, norm_min, norm_max


# In[23]:


from IPython.core.display import display, HTML

def save_ccd_params(save_prefix, mean, stdev, normalize, clip, norm_percentile, prob_thresh,
                    min_intensity, chunk_size, overlap):
    # save parameters to a file
    ccd_param_dict = {
        'mean': mean,
        'stdev': stdev,
        'normalize': normalize,
        'norm_min': norm_min,
        'norm_max': norm_max,
        'clip': clip,
        'norm_percentile': [norm_percentile],
        'prob_thresh': prob_thresh,
        'min_intensity': min_intensity,
        'chunk_size': [chunk_size],
        'overlap': overlap,
        'sigma': sigma,
        'steepness': steepness,
        'offset': offset,
        'min_dist': min_dist
    }
    if ccd_af:
        ccd_param_dict['af_sutraction'] = 1
    else:
        ccd_param_dict['af_sutraction'] = 0
    print("ccd_param_dict: ", ccd_param_dict)
    df = pd.DataFrame(ccd_param_dict)
    ccdp_f = 'ccd_phathom_params_%s.csv'%datatype
    df.to_csv(os.path.join(save_prefix, ccdp_f), sep=',', index=False)
    print("saved ccd params to %s"%(ccdp_f))
    display(HTML(df.style.render()))
    
    return ccdp_f
    
ccd_phathom_params_f = save_ccd_params(os.path.join(workdir, params_byType['rel_path']),
                            mean, stdev, normalize, clip, norm_percentile, prob_thresh,
                            min_intensity, chunk_size, overlap)


# In[24]:


# NOTE:
# as we'll normalize chunks with global min & max inside, 
# there's no need for additional division in intensity_probability()
NUM_CPU=10
centers = BM_CCD.calc_centroids_parallel(zarr,
                                          sigma, steepness, offset,
                                          mean, stdev,
                                          prob_thresh, min_dist,
                                          min_intensity, 
                                          chunk_size, overlap,
                                          nb_workers=NUM_CPU,
                                          normalize=normalize,
                                          normByPercentile=norm_percentile,
                                          clip=clip,
                                          zarr_subtract=af_zarr,
                                          #norm_min=None,
                                          #norm_max=None,
                                          norm_min=norm_min,
                                          norm_max=norm_max,
                                          prob_output=None)


# In[25]:


len(centers)


# In[26]:


n = 0
n_none = 0
n_nn = 0
nc = len(centers)
for c in centers:
    if c is not None:
        n += c.shape[0]
        n_nn += 1
    else:
        n_none += 1
        
print("Total: %d (None: %.2f%%, Not-None:%.2f%%)"%(nc, n_none/float(nc)*100., n_nn/float(nc)*100.))
print("Total # of centers: %d"%n) 


# In[27]:


centers_list = [c for c in centers if c is not None]
len(centers_list)
centers_vstack = np.vstack(centers_list)
centers_vstack.shape


# In[28]:


len(centers_vstack)


# In[29]:


# save to npy
cc_npy_fname = 'cell_centers_%s.npy'%datatype
np.save(os.path.join(workdir, params_byType['rel_path'],cc_npy_fname), centers_vstack)


# In[30]:


# build dictionary
cdict_all = []
for idx, c in enumerate(centers_vstack):
    cdict = {}
    cdict['id'] = idx
    cdict['z'] = c[0]
    cdict['y'] = c[1]
    cdict['x'] = c[2]
    cdict['label'] = -1
    cdict_all.append(cdict)

len(cdict_all)


# In[31]:


# save to csv
df = pd.DataFrame(cdict_all)


# In[32]:


df = df[['id', 'z', 'y', 'x', 'label']]
df.head()


# In[33]:


cc_csv_fname = 'cell_centers_%s.csv'%datatype
df.to_csv(os.path.join(workdir, params_byType['rel_path'],cc_csv_fname),
          sep=',', index=False)


# In[34]:


# update params
params_byType['cc_npy'] = cc_npy_fname
params_byType['cc_csv'] = cc_csv_fname
params_byType['ccd_phathom_param_csv'] = ccd_phathom_params_f
params


# In[35]:


# save back
with open(paramf, 'w') as fp:
    json.dump(params, fp, indent=4, separators=(',', ':'), sort_keys=True)
    fp.write('\n')


# ### Run Sample Cell Center Visualizer

# In[36]:


# See separated notebook (CellCenterDetectionResultVisualization.ipynb)


# In[ ]:




