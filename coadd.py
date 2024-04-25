import numpy as np
from glob import glob
from astropy.io import fits
import time,os
import matplotlib.pyplot as plt
from astropy.io import ascii

def read_header(headPath):
    '''
    Reads in a SCAMP .head file, skipping history and comment lines.
    Currently written to work with multi-extension FITS headers output by
    SCAMP.  These seem to be in the same index order as the image, so no
    need to track by CCD name.

    Parameters
    ----------
    headPath : `string`
        Full or relative path to header file to be read in.

    Returns
    -------
    headers : `dict`
        Dictionary of dictionaries.  Each dictionary contains keywords for one
        header corresponding to one CCD.  32 headers in total, accessed by
        index (0--31).

    '''
    keys = range(32)
    headers = {}
    for key in keys:
        headers[key] = {}

    key = 0
    f = open(headPath)
    for line in f:
        if line.startswith('HISTORY'):
            continue
        elif line.startswith('COMMENT'):
            continue
        elif line.startswith('END'):
            key += 1
        else:
            cols = line.split('=')
            val = cols[1][: cols[1].find('/')]
            try:
                val = float(val)
            except ValueError:
                val = val.strip()
                val = val.strip("'")
            headers[key][cols[0].strip()] = val

    return headers


def rewrite_wcs(imPath, headPath, outputPath):
    '''
    Pastes over the old WCS info in each image header with that derived from
    the external .head file from SCAMP.
    Overwrites the old image with the new one, with the updated WCS header.

    Parameters
    ----------
    imPath : `string`
        Full or partial path to the image you want to modify
    headPath : `string`
        Full or partial path to the SCAMP header file associated with imPath
    outputPath : `string`
        Output image path and name.  Use imPath to overwrite the original.

    Returns
    -------
    None.  Write new image to the disk, with an updated header.

    '''
    headers = read_header(headPath)
    im = fits.open(imPath)
    for i in headers.keys():
        for key in headers[i]:
            im[i+1].header[key] = headers[i][key]
    im.writeto(outputPath, overwrite=True)


def cull_ldac_table(tablePath, outPath):
    '''
    Culls an input FITS_LDAC detection table from SExtractor of non-pointlike
    sources, and write a new table to the disk.

    Parameters
    ----------
    tablePath : `string`
        Full or partial path to SExtractor table to be modified
    outPath : `string`
        Full or partial path to modified output table, to be written to the
        disk.  Set this to tablePath to overwrite the original table.

    Returns
    -------
    None.  Writes a culled table to the disk.
    '''
    #try:
    #    tab = fits.open(tablePath)
    #except OSError:
    #    print 'SExtractor table must be in FITS_LDAC format.'
    #assert 'ELLIPTICITY' in tab[2].data.names, \
    #    print 'Missing key ELLIPTICITY; adjust default.param to include this'
    #assert 'CLASS_STAR' in tab[2].data.names, \
    #    print 'Missing key CLASS_STAR; adjust default.param to include this'

    tab = fits.open(tablePath)
    
    tabIdx = 2
    for i in range((len(tab)-1)/2):
        # Values close to 1 in CLASS_STAR or 0 in ELLIPTICITY are stars
        want = (tab[tabIdx].data['CLASS_STAR'] >= 0.75) \
            & (tab[tabIdx].data['ELLIPTICITY'] <= 0.1) \
            & (tab[tabIdx].data['MAG_AUTO'] <= -11.0)
        # Setting FWHM of bad detections to a value below the choice of
        # threshold in current scamp.conf file (2.0 to 6.0px)
        tab[tabIdx].data['FLUX_RADIUS'][~want] = 1.0
        tabIdx += 2

    tab.writeto(outPath, overwrite=True)

def fix_wcs(frames, prefix = 'w', sex = True, se_para = 'default.sex', sc_para = 'scamp.conf'):
    '''
    Runs sextractor and scamp to fix WCS in a list of images.

    Parameters
    ----------
    frames : `list`
        List of strings containing the filenames of the fits images.
    prefix : `string`
        String prepended to the output filenames.
    sex : `Boolean`
        If False, does not run sextractor and insted looks for a previous
        'output.cat' file in the working directory
    se_para : `string`
        Filename of the sextractor configuration file.
    sc_para : `string`
        Filename of the scamp configuration file.

    Returns
    -------
    None. Writes the images with fixed WCS to disk as prefix + filename.
    '''
    for frame in frames:
        if sex:
            cmd = 'sextractor ' + frame + ' -c ' + se_para
            os.system(cmd)

            cull_ldac_table('output.cat', 'culled_output.cat')

        cmd = 'scamp culled_output.cat' + ' -c ' + sc_para
        os.system(cmd)
            
        rewrite_wcs(frame, 'culled_output.head', prefix + frame)

def combine(frames, config = 'default.swarp', out = 'default'):
    '''
    Combines a list of images using SWarp.

    Parameters
    ----------
    frames : `list`
        List of strings containing the filenames of the fits images.
    config : `string`
        Filename of the swarp configuration file.

    Returns
    -------
    None. SWarp output must be defined in the configuration file.
    '''
    ascii.write([frames], 'tmp.lis', overwrite=True, format='no_header')
    if out != 'default':
        cmd = 'SWarp @tmp.lis -c ' + config + ' -IMAGEOUT_NAME ' + out
    else:
        cmd = 'SWarp @tmp.lis -c ' + config
    os.system(cmd)
    

def pop(frames):
    '''
    Removes the faulty CCD "ESO_CCD_#77" from the multi-extension fits files. 
    '''
    for frame in frames:
        hdul=fits.open(frame)
        chips = [ext.header['EXTNAME'] for ext in hdul[1:]]
        hdul.pop(np.where(np.array(chips) == 'ESO_CCD_#77')[0][0]+1)
        hdul[0].header['HIERARCH ESO DET CHIPS'] -= 1
        hdul.writeto('cleaned_'+frame, overwrite = True)
    

def blank(ima):
    '''
    Changes zeros in an image to NaN.
    '''
    hdul = fits.open(ima)
    hdul[0].data[hdul[0].data == 0] = np.nan
    hdul.writeto(ima, overwrite=True)


if __name__ == '__main__':
    start_time = time.time()

    tzo_files = np.array(glob('tzoOMEGA*fits'))
    hdrs = [fits.getheader(frame, output_verify = 'ignore') for frame in tzo_files]
    tzo_ha = []
    tzo_r = []
    for i in range(len(hdrs)):
        if hdrs[i]['HIERARCH ESO INS FILT1 NAME'] == 'NB_659' and hdrs[i]['HIERARCH ESO DPR TYPE'] == 'OBJECT':
            tzo_ha.append(tzo_files[i])
        elif hdrs[i]['HIERARCH ESO INS FILT1 NAME'] == 'r_SDSS' and hdrs[i]['HIERARCH ESO DPR TYPE'] == 'OBJECT':
            tzo_r.append(tzo_files[i])

    sFtzo_ha = 'sF' + np.char.array(tzo_ha)
    sFtzo_r = 'sF' + np.char.array(tzo_r)
    wsFtzo_ha = 'wsF' + np.char.array(tzo_ha)
    wsFtzo_r = 'wsF' + np.char.array(tzo_r)

    print('--- Fixing WCS for Halpha images ---')
    
    fix_wcs(sFtzo_ha, se_para = 'default.sex')
    fix_wcs(wsFtzo_ha, prefix = '', se_para = 'default.sex')


    print('--- Fixing WCS for r-band images ---')

    fix_wcs(sFtzo_r, se_para = 'default_r.sex')
    fix_wcs(wsFtzo_r, prefix = '', se_para = 'default_r.sex')

    clean_ha = 'cleaned_wsF' + np.char.array(tzo_ha)
    clean_r = 'cleaned_wsF' + np.char.array(tzo_r)

    print('--- Combining Halpha images ---')

    pop(wsFtzo_ha)
    combine(clean_ha, 'ha.swarp')
    blank('coadd_ha.fits')

    print('--- Combining r-band images ---')

    pop(wsFtzo_r)
    combine(clean_r, 'r.swarp')
    blank('coadd_r.fits')
    
    print('--- overall runtime: %s seconds' % (time.time() - start_time))
    
    
        
