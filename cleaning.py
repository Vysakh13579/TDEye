import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import json
import matplotlib.gridspec as gridspec

import os
from numpyencoder import NumpyEncoder





def get_lightcurve_data(tde_name = 'ASASSN-14li'):
	"""
	Input: 
		The TDEs name

	Returns:
		1. A dictionary with all of the light curve data, labelled by observing band. 
		2. A list of lightcurve filters with available data. 
	"""

	fname = 'manyTDE/data/sources/{0}.json'.format(tde_name)
	tde_data = json.load(open(fname,'r'))# Load data. 

	# These conversion are needed because json doesn't store tuples.
	dt = [tuple(x) for x in tde_data['lightcurve']['dtype']]
	lc_obj = [tuple(x) for x in tde_data['lightcurve']['data']] 

	# Make a recarray. 
	lc_rec = np.array(lc_obj, dtype=dt)
	mjd0 = tde_data['peak_mjd'] + 2400000.5

	lc_dict = {}
	filters = tde_data['lightcurve']['filters']
	frequency_Hz = tde_data['lightcurve']['frequency_Hz']

	for flt in filters:
		idx = lc_rec['filter']==flt

		flux = lc_rec[idx]['flux_Jy']*1e6
		try:
			flux_corr = flux / tde_data['extinction']['linear_extinction'][flt]# Correct for extinction. 
		except KeyError as err:
			pass
		lc_dict[flt] = [lc_rec[idx]['mjd'] + + 2400000.5, flux_corr, lc_rec[idx]['e_flux_Jy']*1e6]
	
	return lc_dict, filters, frequency_Hz , mjd0



def mag2flux(mag):
    '''
    AB magnitude to Jansky
    '''
    return 10**(-0.4*(mag + 48.6)) *1e23


def q_cuts(ZTF, softcuts=True,output=False):
    '''
    update for Jul 2023
    main difference:  forcediffimfluxap/forcediffimflux less strict
    '''
    
    DC2Jy = mag2flux(ZTF['zpdiff'])

    app_psf_diff = (ZTF['forcediffimfluxap'] - ZTF['forcediffimflux']) #/ np.abs(ZTF['forcediffimfluxap'])

    iok =   (ZTF['sciinpseeing'] < 4.0) * \
            (ZTF['zpmaginpscirms'] < 0.05)  * \
            (ZTF['infobitssci'] < 33554432) * \
            (ZTF['scisigpix'] <= 25) * \
            (ZTF['procstatus'] == 0)  #added by Tim
                
          
    if softcuts==False: 
        iok *=  (ZTF['adpctdif1'] < 0.2)  *\
                (np.abs(app_psf_diff)<200)*\
                (DC2Jy*ZTF['forcediffimflux'] > -50e-6) *\
                (DC2Jy*np.abs(ZTF['forcediffimfluxunc']) < 30e-6)    

    # new in 2022 check for zero-point outliers, per filter
    for flt in np.unique(ZTF['filter']):
        iflt = ZTF['filter']==flt

        if softcuts:
            iok.loc[iflt] *= np.abs(np.log10(DC2Jy.loc[iflt]/np.median(DC2Jy.loc[iflt])))<0.4 
        else:
            iok.loc[iflt] *= np.abs(np.log10(DC2Jy.loc[iflt]/np.median(DC2Jy.loc[iflt])))<0.1 # new 2023 (was 0.4)
    if output:
        # hack for checking cuts
        plt.hist(app_psf_diff,range=[-300,300],bins=30)
        plt.pause(0.1)
        # key = input()

        print ('# of raw points     :', len(ZTF))    
        print ('# of points rejected:', len(ZTF)-sum(iok))

    return iok

def flux_unc_val(ZTF,output=False):
    chisq = ZTF['forcediffimchisq']
    median_chi = np.median(chisq) #instead of mean, more robust

    unc = np.array(ZTF['forcediffimfluxunc']) 
    unc *= np.sqrt(median_chi)

    if output:
        print(median_chi)

        fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(8,5))
        plt.suptitle(f"In {ZTF['filter'].iloc[0]} filter ({len(ZTF)} points)",fontsize=16)
        ax1.set_ylabel(r"$\chi^2$")
        ax1.set_xlabel("Flux")
        ax2.set_xlabel("Counts")
        ax1.scatter(ZTF['forcediffimflux'],chisq)
        ax2.hist(chisq,bins=30,orientation='horizontal')
        plt.show()

    return unc

def clean_data(datapath,ZTF_ID,savepath=None,verbose=False, log_dict_bool = False):
    """
    Cleans ZTF batch request data using the qcuts function. The removed data is neatly logged on a per filter - per field basis. The cleaning
    now works in such a way that only the data from the primary field (defined as the field in which most measurements were made) is used.
    The uncertainty on the flux measurements is validated and subsequently updated following the method of the ZFPS user guide: 
    https://web.ipac.caltech.edu/staff/fmasci/ztf/zfps_userguide.pdf. The cleaned data contains: time in jd, forced difference image PSF-fit flux [DN],
      the 1-sigma uncertainty in the forced difference image PSF-fit flux [DN], the photometric zeropoint for difference image [mag] and the filter 
      (one of ZTF_g, ZTF_r and ZTF_i). The cleaning log contains, for every field in every filter, whether there are even viable measurements and if
      there are if the field in question is the primary field, what the median zeropoint is, what the standard deviation of the zeropoints is, how
      many of the data points were removed in cleaning and the median chi-square of the datapoints before cleaning. For every filter the median 
      chi-square after cleaning is saved only for the primary field - this will differ only slightly from the median chi-square before cleaning. 
      Both cleaned data and cleaning log are saved as json files in the form "(ZTF_ID)_clean_data.json" and "(ZTF_ID)_clean_log.json".
      
      IMPORTANT: dependencies are the numpy, pandas, json and os packages as well as a special json NumpyEncoder by Hunter M. Allen (https://pypi.org/project/numpyencoder/).

    Args:
        datapath (str): Path to the raw data in the form path-to-data-batchf_reqxxxxxxxxxxxxxxxxx_lc.txt.
        ZTF_ID (str): ZTF identifier of the transient.
        savepath (str, optional): Path to folder in which clean data and cleaning log will be saved. Defaults to None, in which case data is printed if verbose is True, otherwise it is lost.
        verbose (bool, optional): Controls the (amount of) print statements in the function. Defaults to False.
    """
    #Read in the raw data from the data path as a Pandas DataFrame. 
    columns = ['sindex', 'field', 'ccdid', 'qid', 'filter', 'pid', 'infobitssci', 'sciinpseeing', 'scibckgnd', 'scisigpix', 'zpmaginpsci', 'zpmaginpsciunc', 'zpmaginpscirms', 'clrcoeff', 'clrcoeffunc', 'ncalmatches', 'exptime', 'adpctdif1', 'adpctdif2', 'diffmaglim', 'zpdiff', 'programid', 'jd', 'rfid', 'forcediffimflux', 'forcediffimfluxunc', 'forcediffimsnr', 'forcediffimchisq', 'forcediffimfluxap', 'forcediffimfluxuncap', 'forcediffimsnrap', 'aperturecorr', 'dnearestrefsrc', 'nearestrefmag', 'nearestrefmagunc', 'nearestrefchi', 'nearestrefsharp', 'refjdstart', 'refjdend', 'procstatus']
    dtypes = [(columns[x],float) for x in range(len(columns))]
    dtypes[4] = ('filter',r'U8')
    data = pd.DataFrame(np.genfromtxt(datapath,skip_header=53,dtype=dtypes))

    clean_data_full = pd.DataFrame() #an empty frame on which the data from every filter will be vertically stacked.
    iok = q_cuts(data, softcuts=True) #very first quality check, mask array of good data points.
    data_ok = data[iok]
    logdict = {} #dictionary that will form the log.json file

    filters = np.unique(data['filter'])
    filtermasks = [data['filter'] == f for f in filters]
    # fields,field_counts = np.unique(data['field'],return_counts=True) #return_counts for picking the primary field
    fields,_ = np.unique(data_ok['field'],return_counts=True) #Do we want the primary field on data or on data_ok?
    fieldmasks = [data['field'] == fid for fid in fields]

    if iok.sum() == 0: #this might occur, this prevents an error
        print(f"{ZTF_ID}: no viable data found in the batch request. Proceeding to next file.")
        return 0.0
    
    for i,filter in enumerate(filters):
            logdict[filter] = {}
            logdict[filter]["no_viable_data"] = 0 #can be used for a check when loading in the data; if this is True then the data is useless in this particular filter 
            filtermask = filtermasks[i]
            iok_filter = (iok * filtermask) #this checks if something is ok according to qcuts and is in a certain filter. Has the same len as data.
            
            if iok_filter.sum() == 0: #this might occur, this prevents an error
                #print(f"{ZTF_ID}: no viable data found in {filter}. Proceeding to next filter.")
                logdict[filter]["no_viable_data"] = 1
                continue 

            #If we take the primary field on a per filter basis use three lines below.
            data_ok_filter = data[iok_filter]
            filter_field_counts = [np.sum(data_ok_filter['field'] == fid) for fid in fields] #count for each field we know to have in the uncleaned data how often it appears in this filter. Might yield 0's! 
            primary_field = [c == np.max(filter_field_counts) for c in filter_field_counts] #should be this one if we want to pick the primary on a per filter basis

            for j,fid in enumerate(fields):
                field_mask = fieldmasks[j]
                iok_filter_field = iok_filter * field_mask #this checks if something is ok according to qcuts, is in a certain filter and is in a certain field. Has the same len as data.

                logdict[filter][fid] = {}
                if iok_filter_field.sum() == 0: #this might occur, this prevents an error
                    #print(f"{ZTF_ID}: no viable data found in field {fid} of filter {filter}. Proceeding to next field for this filter.")
                    logdict[filter][fid]["no_viable_data"] = 1 #can be used for a check when loading in the data; if this is True then the data is useless in this particular field / filter combo
                    continue

                data_ok_filter_field = data[iok_filter_field]
                data_filter_field = data[filtermask*field_mask] #this is the uncleaned data of this field in this filter
                zeropoint = data_ok_filter_field['zpdiff'].values

                logdict[filter][fid] = {"primary_field":int(primary_field[j]),
                                        "ccdid" : data_ok_filter_field.ccdid.unique(),
                                        "qid" : data_ok_filter_field.qid.unique(),
                                        "median_zeropoint":np.median(zeropoint),'std_zeropoint':np.std(zeropoint),
                                        "removed_in_cleaning":np.sum(np.invert(iok_filter_field)),
                                        "amount_before_cleaning": len(iok_filter_field),
                                        "median_chi2":np.median(data_filter_field['forcediffimchisq']),
                                            "no_viable_data":0}

                if primary_field[j]:
                    #correct the errors of the clean data in this filter (only on the primary field)
                    new_unc = np.array(flux_unc_val(data_ok_filter_field))
                    #the median chi squared after is that of the good data in the primary field of the respective filter
                    logdict[filter]["median_chi2_after"] = np.median(data_ok_filter_field['forcediffimchisq']) 
                    clean_data_filt = pd.DataFrame({'time':data_ok_filter_field['jd'], 'flux': mag2flux(data_ok_filter_field['zpdiff']) * data_ok_filter_field['forcediffimflux'] * 10**6, 
                                                    'fluxDN':data_ok_filter_field['forcediffimflux'],
                                                    'flux_unc': mag2flux(data_ok_filter_field['zpdiff']) * np.abs(data_ok_filter_field['forcediffimfluxunc']) * 10**6, 
                                                   'flux_uncDN':new_unc,'zeropoint':data_ok_filter_field['zpdiff'],
                                                   'filter':data_ok_filter_field['filter']})
                    clean_data_full = pd.concat([clean_data_full,clean_data_filt],ignore_index=True)
        

        
    if savepath != None:
        # clean_data_full.to_json(os.path.join(savepath,str(ZTF_ID)+'_clean_data.json'))
        clean_data_full.to_csv(os.path.join(savepath,str(ZTF_ID)+'_clean_data.txt'),sep='\t',index=None,header=None,mode='a')
        with open(os.path.join(savepath,str(ZTF_ID)+'_clean_log.json'),'w') as outfile:
            json.dump(logdict,outfile,indent=4,ensure_ascii=False,separators=(',',':'),cls=NumpyEncoder)
        #print('No savepath provided. Dumping results, shown if verbose set to True.')
        if log_dict_bool:
            return clean_data_full, logdict
        else:
            return clean_data_full
    else:
        #print('No savepath provided. Dumping results, shown if verbose set to True.')
        if log_dict_bool:
            return clean_data_full, logdict
        else:
            return clean_data_full
        if verbose:
            print(clean_data_full.to_markdown())
            print()
            print(logdict)


def inverse_variance_weighting(values, sigma):

    values = np.array(values)
    variances = np.array(sigma**2)
    weights = 1 / variances
    weighted_average = np.sum(weights * values) / np.sum(weights)
    weighted_variance = 1 / np.sum(weights)

    return weighted_average, np.sqrt(weighted_variance)

def baseline_correction(data, goback = 100, threshold_obs = 30, unit = 'uJy', peak_time = 0):
    df = data.copy(deep=True)
    filters = np.unique(df['filter'])
    df['baseline_crctd_flux'] = -9999
    df['baseline_crctd_flux_unc'] = -9999
    crctn_data = {}
    for filt in filters:
        time =df.loc[df['filter'] == filt].time.to_numpy()
        if unit == 'uJy':
            flux =  df.loc[df['filter'] == filt].flux.to_numpy()
            fluxerr = df.loc[df['filter'] == filt].flux_unc.to_numpy()
        elif unit == 'DN':
            flux =  df.loc[df['filter'] == filt].fluxDN.to_numpy()
            fluxerr = df.loc[df['filter'] == filt].flux_uncDN.to_numpy()

        # peak_index = np.argmax(flux)
        len_gobackd = sum(time<(peak_time - goback))
       
        if len_gobackd>threshold_obs:
            flux_base = flux[time<(peak_time - goback)]
            flux_unc_base = fluxerr[time<(peak_time - goback)]
            avg, avg_unc = inverse_variance_weighting(flux_base, flux_unc_base)
            df.loc[df['filter'] == filt,'baseline_crctd_flux'] = flux - avg
            df.loc[df['filter'] == filt,'baseline_crctd_flux_unc'] = np.sqrt(fluxerr**2 + avg_unc**2)

            
            crctn_data[filt + '_offset'] = avg
            crctn_data[filt + '_used_points'] = len_gobackd
        else:
            crctn_data[filt + '_used_points'] = len_gobackd
    return df, crctn_data
