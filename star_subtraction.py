#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Program for subtracting stars and sky from flattened OMEGACAM fits images.
"""
import numpy as np
from glob import glob
from astropy.io import fits, ascii
from astropy.wcs import WCS
import time,os
from scipy.interpolate import NearestNDInterpolator
from scipy import ndimage
import coadd

def cp_headers(frames, donor_frames, prefix = ''):
    '''
    Copies the headers from donor frames to the input frames. Usefull for 
    transfering WCS solutions.
    '''
    for i in range(len(frames)):
        hdul = fits.open(frames[i], output_verify = 'ignore')
        chips = [ext.header['EXTNAME'] for ext in hdul[1:]]
        for chip in chips:
            new_hdr = fits.getheader(donor_frames[i], chip)
            hdul[chip].header = new_hdr
        hdul.writeto(prefix + frames[i], overwrite=True)

def subtract_stars(frames, psf, star_catalog, prefix = 'p', overwrite = True):
    '''
    Subtracts stars from a list of frames. Works by scaling PSF acording to
    a catalog scale factors to each star, and subtracting them from all CCDs.
    '''
    start_time = time.time()
    starcat = fits.getdata(star_catalog)
    for frame in frames:
        if not overwrite and os.path.exists(prefix + frame):
              print('--- Skipping ' + prefix + frame  + ' as it already exists ---')
              continue
        hdul = fits.open(frame, output_verify = 'ignore')
        chips = [ext.header['EXTNAME'] for ext in hdul[1:]]
        for chip in chips:
            if chip == 'ESO_CCD_#77':
                continue
            w = WCS(hdul[chip].header)
            xcen, ycen = w.all_world2pix(starcat['ra'],starcat['dec'],1)
            inGNU = frame
            hduGNU = ' --hdu=' + chip
            for i in range(len(xcen)):
                x1 = xcen[i] - 3500
                x2 = xcen[i] + 3500
                y1 = ycen[i] - 3500
                y2 = ycen[i] + 3500
                # choose only stars whose PSF overlaps with the CCD
                if x1 < 2048 and x2 > 0 and y1 < 4100 and y2 > 0:
                    center = str(starcat['ra'][i]) + ',' + \
                             str(starcat['dec'][i])
                    scale = str(starcat['scale_factor'][i])
                    outGNU = 'tmp.fits'
                    cmd = 'astscript-psf-subtract ' + inGNU + ' --mode=wcs ' + \
                            '--psf=' + psf + ' --scale=' + scale + \
                            ' --center=' + center + ' --output=' + outGNU + \
                            hduGNU
                    os.system(cmd)
                    inGNU = chip + '.fits'
                    hduGNU = ''
                    os.system('cp ' + outGNU + ' ' + inGNU)
                    os.system('rm tmp.fits')
            if os.path.exists(chip + '.fits'):
                hdul[chip].data = fits.getdata(chip + '.fits')
                os.system('rm ' + chip + '.fits')
        filename = prefix + frame
        hdul.writeto(filename, output_verify = 'ignore', overwrite = True)
        hdul.close()
        print('--- overall runtime: %s seconds' % (time.time() - start_time))
                    

def subtract_sky(frames, mask_frames, bpm = None, vmap = None, bcm = True):
    '''
    Subtracts sky in a list of frames. Works CCD by CCD by calculating 
    the mode in each masked frame within one OB, then scaling these modes to
    the mode of one of the frames, and taking the median of the scaled modes as
    the sky value for that CCD in that frame.

    Parameters
    ----------
    frames : `list`
        List of strings containing the filenames of the fits images.
    mask_frames : `list`
        List of strings containing the filenames of the binary masks 
    bpm : `string`
        Filename of a bad pixel map.
    vmap : `string`
        Filename of a vignetting map.

    Returns
    -------
    None. Writes the sky-subtracted images to the disk as s + 'frame'.
    '''
    # remove any loose temp files
    os.system('rm temp_masked_*fits')

    # get the OB name for each frame
    OBs = []
    for frame in frames:
        hdr = fits.getheader(frame)
        OB = hdr['HIERARCH ESO OBS name']
        date = hdr['DATE']
        if OB == 'ha_8' and date[8:10] == '16':
            OBs.append(OB + '1')
        elif OB == 'ha_8' and date[8:10] == '17':
            OBs.append(OB + '2')
        elif OB == 'r_8' and date[8:10] == '16':
            OBs.append(OB + '1')
        elif OB == 'r_8' and date[8:10] == '17':
            OBs.append(OB + '2')
        else:
            OBs.append(OB)
    OBs = np.array(OBs)

    # get a list of unique OBs and loop over them
    unique_OBs = list(dict.fromkeys(OBs))
    for OB in unique_OBs:
        frames_OB = frames[OBs == OB] # subarray of frames within this OB
        masks_OB = mask_frames[OBs == OB] # subarray of masks within this OB

        # open mask and image hduls for this OB
        hduls_ima = [fits.open(f, output_verify = 'ignore') for f in frames_OB]
        hduls_mask = [fits.open(f, output_verify = 'ignore') for f in masks_OB]

        # create temp_masked* files from frames with masked pixels set to NaN
        chips = [ext.header['EXTNAME'] for ext in hduls_ima[0][1:]]
        if bcm:
            chips.remove('ESO_CCD_#77')
        for i in range(len(hduls_ima)):
            for chip in chips:
                masked = hduls_ima[i][chip].data * 1.
                masked[hduls_mask[i][chip].data == 1] = np.NaN
                if bpm:
                    badpix = fits.getdata(bpm, chip)
                    masked[badpix == 1] = np.NaN
                if vmap:
                    vignetted = fits.getdata(vmap, chip)
                    masked[vignetted == 1] = np.NaN
                hduls_mask[i][chip].data = masked # writing the masked data over the mask to save memory
                                
            hduls_mask[i].writeto('temp_masked_'+frames_OB[i], output_verify = 'ignore', overwrite = True)
            hduls_mask[i].close() # closing masks as we don't need them anymore

        # for each CCD, count the background mode
        modes = []
        for i in range(len(hduls_mask)):
            modes.append([])
            for chip in chips:
                cmd = 'aststatistics temp_masked_' + frames_OB[i] + ' -mode --hdu=' + chip + ' > temp.txt'
                os.system(cmd)
                with open('temp.txt') as f:
                    modes[-1].append(float(f.read()[:-1]))

        modes = np.array(modes)
        ima_mm = np.median(modes, 1)

        # for each CCD, scale the mode of the other images in the OB to the mode of this one, then calculate the median of those modes and subtract it from the image
        for j in range(len(chips)):
            for i in range(len(hduls_ima)):
                mmm = []
                for k in range(len(hduls_ima)):
                    mmm.append(modes[k][j] * ima_mm[i] / ima_mm[k])
                sky = np.median(mmm)
                hduls_ima[i][chips[j]].data -= sky

        # cleanup and writing sky-subtracted image to disk
        ascii.write(modes, OB+'_sky.tab', names = chips, overwrite = True)
        os.system('rm temp_masked_*fits')
        for i in range(len(hduls_ima)):
            filename = 's' + frames_OB[i]
            hduls_ima[i].writeto(filename, output_verify = 'ignore', overwrite = True)
            hduls_ima[i].close()
            print('Wrote deskied image to disk as ' + filename)


def bin_and_interp(hdulist, block=9, hdu=1):
    '''
    Median bins image into block x block size, then interpolates flux using
    across masked pixels using nearest neighbors approach.
    Requires:
        --Image HDUList object, so the output of fits.open()
        --Block factor (or if none given, defaults to 9px x 9px)
    Returns:
        --Binned image resized to standard image size (no header)
    ''' 
    # Shaving off excess pixels given bin size
    xedge = np.shape(hdulist[hdu].data)[0] % block
    yedge = np.shape(hdulist[hdu].data)[1] % block
    imtrim = hdulist[hdu].data.copy()
    imtrim = imtrim[xedge:, yedge:]
        
    # Reshape image array into arrays of block x block
    binim = np.reshape(imtrim,
                       (np.shape(imtrim)[0]//block,
                        block,
                        np.shape(imtrim)[1]//block,
                        block)
                       )
    binim[binim == -999.] = np.nan
        
    # Have to keep bins with very few masked pixels from skewing the results
    binned = np.zeros((np.shape(imtrim)[0]//block, np.shape(imtrim)[1]//block))
    for i in range(binim.shape[0]):
        for j in range(binim.shape[2]):
            box = binim[i, : , j , :]
            msk = np.isfinite(box)
            if len(msk[msk]) <= 30:
                binned[i,j] = np.nan
            else:
                binned[i,j] = np.nanmedian(box)
    binned[np.isnan(binned)] = -999.0
        
    # Interpolate flux across masks using nearest neighbors
    bnmsk = binned!=-999
    X, Y = np.meshgrid(np.arange(binned.shape[1]), np.arange(binned.shape[0]))
    xym = np.vstack( (np.ravel(X[bnmsk]), np.ravel(Y[bnmsk])) ).T
    data = np.ravel(binned[bnmsk])
    interp = NearestNDInterpolator(xym, data)
    binned = interp(np.ravel(X), np.ravel(Y)).reshape(X.shape)
        
    # Big version of binned image
    for i in range(binim.shape[0]):
        for j in range(binim.shape[2]):
            imtrim[i*block : (i+1)*block, j*block : (j+1)*block] = binned[i, j]
                
    # set edge pixels to a median value if size % block != 0
    bigbin = np.zeros(hdulist[hdu].data.shape) + np.median(imtrim) 
    bigbin[xedge:, yedge:] = imtrim
    
    return bigbin
            

def subtract_complex_sky(frames, mask_frames, bpm = None, vmap = None, bcm = True, block=128, prefix = 's', overwrite = True):
    '''
    Subtracts sky in a list of frames. Works CCD by CCD by calculating 
    the mode in each masked frame within one OB, then scaling these modes to
    the mode of one of the frames, and taking the median of the scaled modes as
    the sky value for that CCD in that frame.

    Parameters
    ----------
    frames : `list`
        List of strings containing the filenames of the fits images.
    mask_frames : `list`
        List of strings containing the filenames of the binary masks 
    bpm : `string`
        Filename of a bad pixel map.
    vmap : `string`
        Filename of a vignetting map.

    Returns
    -------
    None. Writes the sky-subtracted images to the disk as s + 'frame'.
    '''
    # remove any loose temp files
    os.system('rm temp_masked_*fits')

    # get the OB name for each frame
    OBs = []
    for frame in frames:
        hdr = fits.getheader(frame)
        OB = hdr['HIERARCH ESO OBS name']
        date = hdr['DATE']
        if OB == 'ha_8' and date[8:10] == '16':
            OBs.append(OB + '1')
        elif OB == 'ha_8' and date[8:10] == '17':
            OBs.append(OB + '2')
        elif OB == 'r_8' and date[8:10] == '16':
            OBs.append(OB + '1')
        elif OB == 'r_8' and date[8:10] == '17':
            OBs.append(OB + '2')
        else:
            OBs.append(OB)
    OBs = np.array(OBs)

    # get a list of unique OBs and loop over them
    unique_OBs = list(dict.fromkeys(OBs))
    for OB in unique_OBs:
        frames_OB = frames[OBs == OB] # subarray of frames within this OB
        masks_OB = mask_frames[OBs == OB] # subarray of masks within this OB

        if not overwrite and os.path.exists(prefix + frames_OB[0]):
            print('--- Skipping OB ' + OB + ' as ' + prefix + frames_OB[0]  + ' already exists ---')
            continue
        # open mask and image hduls for this OB
        hduls_ima = [fits.open(f, output_verify = 'ignore') for f in frames_OB]
        hduls_mask = [fits.open(f, output_verify = 'ignore') for f in masks_OB]

        # create temp_masked* files from frames with masked pixels set to NaN
        chips = [ext.header['EXTNAME'] for ext in hduls_ima[0][1:]]
        if bcm:
            chips.remove('ESO_CCD_#77')
        for i in range(len(hduls_ima)):
            for chip in chips:
                masked = hduls_ima[i][chip].data * 1.
                masked[hduls_mask[i][chip].data == 1] = np.NaN
                if bpm:
                    badpix = fits.getdata(bpm, chip)
                    masked[badpix == 1] = np.NaN
                if vmap:
                    vignetted = fits.getdata(vmap, chip)
                    masked[vignetted == 1] = np.NaN
                hduls_mask[i][chip].data = masked # writing the masked data over the mask to save memory
            hduls_mask[i].writeto('temp_masked_'+frames_OB[i], output_verify = 'ignore', overwrite = True)

        # create a sky image for each CCD by scaling, median combining, binning and smoothing the masked images
        modes = []
        for chip in chips:
            modes.append([])
            for i in range(len(hduls_ima)):
                w = WCS(hduls_ima[i][chip].header)
                galpix = w.all_world2pix(204.253958,-29.865417,1)
                if galpix[1] > -1400 and galpix[1] < 5500 and galpix[0] > -2400 and galpix[0] < 4448:
                    modes[-1].append(np.nan)
                else:
                    cmd = 'aststatistics temp_masked_' + frames_OB[i] + ' -mode --hdu=' + chip + ' > temp.txt'
                    os.system(cmd)
                    with open('temp.txt') as f:
                        modes[-1].append(float(f.read()[:-1]))
                                        
        modes = np.array(modes)
        ima_mm = np.nanmedian(modes, 0)
        for j in range(len(chips)):
            chip = chips[j]
            num = 0
            for i in range(len(hduls_ima)):
                if not np.isnan(modes[j][i]):
                    hduls_mask[i][chip].data *= ima_mm[0] / ima_mm[i]
                    fits.writeto('temp_sky_' + str(i) + '.fits', hduls_mask[i][chip].data, output_verify = 'ignore', overwrite = True)
                    num += 1

            cmd = 'astarithmetic temp_sky*fits ' + str(num) + ' 3 0.2 sigclip-median -g0 --output=' + chip + '_sky.fits'
            os.system(cmd)
            os.system('rm temp_sky_*fits')
            stack = fits.open(chip + '_sky.fits')
            stack[1].data = bin_and_interp(stack, block=block)
            stack[1].data = ndimage.gaussian_filter(stack[1].data, block//2)
            stack.writeto(chip + '_sky.fits', overwrite=True)
            stack.close()

        for i in range(len(hduls_ima)):
            hduls_mask[i].close() # closing masks as we don't need them anymore

        # for each CCD, scale the sky image to it and subtract the sky from it
        for j in range(len(chips)):
            for i in range(len(hduls_ima)):
                sky_ima = fits.getdata(chips[j] + '_sky.fits')
                #print(frames_OB[i], chips[j], ima_mm[i] / ima_mm[0])
                sky_ima *= ima_mm[i] / ima_mm[0]
                hduls_ima[i][chips[j]].data -= sky_ima
                

        # cleanup and writing sky-subtracted image to disk
        ascii.write(modes.T, OB+'_sky.tab', names = chips, overwrite = True)
        #os.system('rm temp_masked_*fits')
        for i in range(len(hduls_ima)):
            filename = prefix + frames_OB[i]
            hduls_ima[i].writeto(filename, output_verify = 'ignore', overwrite = True)
            hduls_ima[i].close()
            print('Wrote deskied image to disk as ' + filename)


def skycor(frames, mask_frames, bpm = None, vmap = None):
    for i in range(len(frames)):
        hdul = fits.open(frames[i])
        mask = fits.open(mask_frames[i])
        chips = [ext.header['EXTNAME'] for ext in hdul[1:]]
        chips.remove('ESO_CCD_#77')
        med = []
        for chip in chips:
            masked = hdul[chip].data * 1.
            masked[mask[chip].data == 1] = np.NaN
            if bpm:
                badpix = fits.getdata(bpm, chip)
                masked[badpix == 1] = np.NaN
            if vmap:
                vignetted = fits.getdata(vmap, chip)
                masked[vignetted == 1] = np.NaN
            med.append(np.nanmedian(masked))
        ima_mm = np.nanmedian(med)
        for chip in chips:
            hdul[chip].data -= ima_mm
        hdul.writeto(frames[i], overwrite = True)

def subtract_continuum(frames_on, frames_off, config = 'tst.swarp', scale_factor = 0.82, prefix = 'c'):
    for i in range(len(frames_on)):
        hdul_off = fits.open(frames_off[i])
        chips = [ext.header['EXTNAME'] for ext in hdul_off[1:]]
        for chip in chips:
            hdul_off[chip].data *= scale_factor * (-1)
        hdul_off.writeto('tmp_off.fits', overwrite = True)
        ascii.write([[frames_on[i], 'tmp_off.fits']], 'tmp.lis', overwrite = True, format = 'no_header')
        out = prefix + frames_on[i]
        cmd = 'SWarp @tmp.lis -c ' + config + ' -IMAGEOUT_NAME ' + out
        os.system(cmd)
        #os.system('rm tmp_off.fits')
        #os.system('rm tmp.lis')
        
def cull_bad_chips(frames, chiplist, prefix=''):
    for frame in frames:
        hdul = fits.open(frame)
        chips = [ext.header['EXTNAME'] for ext in hdul[1:]]
        for chip in chiplist:
            hdul.pop(np.where(np.array(chips) == chip)[0][0]+1)
            chips.remove(chip)
            hdul[0].header['HIERARCH ESO DET CHIPS'] -= 1
        hdul.writeto(prefix + frame, overwrite = True)


if __name__ == '__main__':
    start_time = time.time()

    tzo_files = glob('tzoOMEGA*fits')
    tzo_files.remove('tzoOMEGA.2021-01-21T06:38:04.356.fits')
    tzo_files.remove('tzoOMEGA.2021-01-21T06:28:09.564.fits')
    tzo_files.remove('tzoOMEGA.2021-01-21T06:18:15.063.fits')
    tzo_files.remove('tzoOMEGA.2021-01-21T06:08:20.202.fits')
    tzo_files.remove('tzoOMEGA.2021-01-21T06:02:07.831.fits')
    tzo_files.remove('tzoOMEGA.2021-01-21T06:00:28.110.fits')
    tzo_files.remove('tzoOMEGA.2021-01-21T05:58:48.660.fits')
    tzo_files.remove('tzoOMEGA.2021-01-21T05:57:09.220.fits')
    tzo_files = np.array(tzo_files)
    hdrs = [fits.getheader(frame, output_verify = 'ignore') for frame in tzo_files]
    tzo_ha = []
    tzo_r = []
    for i in range(len(hdrs)):
        if hdrs[i]['HIERARCH ESO INS FILT1 NAME'] == 'NB_659' and hdrs[i]['HIERARCH ESO DPR TYPE'] == 'OBJECT':
            tzo_ha.append(tzo_files[i])
        elif hdrs[i]['HIERARCH ESO INS FILT1 NAME'] == 'r_SDSS' and hdrs[i]['HIERARCH ESO DPR TYPE'] == 'OBJECT':
            tzo_r.append(tzo_files[i])

    Ftzo_ha = 'F' + np.char.array(tzo_ha)
    Ftzo_r = 'F' + np.char.array(tzo_r)
    wsFtzo_ha = 'wsF' + np.char.array(tzo_ha)
    wsFtzo_r = 'wsF' + np.char.array(tzo_r)

    print('--- Copying fixed WCS (Halpha) ---')

    cp_headers(Ftzo_ha, wsFtzo_ha)
    
    print('--- Copying fixed WCS (r-band) ---')
    
    cp_headers(Ftzo_r, wsFtzo_r)

    print('--- Subtracting stars (Halpha) ---')

    subtract_stars(Ftzo_ha, 'psf_ha_skysub_full_new.fits', 'starcat_ha.fits')

    print('--- Subtracting stars (r-band) ---')

    subtract_stars(Ftzo_r, 'psf_r_skysub_full_new.fits', 'starcat_r.fits')

    pFtzo_ha = 'pF' + np.char.array(tzo_ha)
    pFtzo_r = 'pF' + np.char.array(tzo_r)
    masks_ha = 'mask_f' + np.char.array(tzo_ha)
    masks_r = 'mask_f' + np.char.array(tzo_r)
    SpFtzo_ha = 'SpF' + np.char.array(tzo_ha)
    SpFtzo_r = 'SpF' + np.char.array(tzo_r)

    print('--- Subtracting sky (Halpha) ---')

    subtract_complex_sky(pFtzo_ha, masks_ha, bpm = 'BPM_ha.fits', vmap = 'ha_cross_leak.fits', prefix='S')
    skycor(SpFtzo_ha, masks_ha, bpm = 'BPM_ha.fits', vmap = 'ha_cross_leak.fits')

    print('--- Subtracting sky (r-band) ---')

    subtract_complex_sky(pFtzo_r, masks_r, bpm = 'BPM_r.fits', prefix='S')
    skycor(SpFtzo_r, masks_r, bpm = 'BPM_r.fits')

    clean_ha = 'cleaned_SpF' + np.char.array(tzo_ha)
    clean_r = 'cleaned_SpF' + np.char.array(tzo_r)

    print('--- Combining Halpha images ---')

    coadd.pop(SpFtzo_ha)
    coadd.combine(clean_ha, 'ha.swarp')
    coadd.blank('coadd_ha.fits')

    print('--- Combining r-band images ---')

    coadd.pop(SpFtzo_r)
    coadd.combine(clean_r, 'r.swarp')
    coadd.blank('coadd_r.fits')

    
    print('--- overall runtime: %s seconds' % (time.time() - start_time))
