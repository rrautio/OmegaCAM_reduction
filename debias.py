#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Program for debiasing OMEGACAM fits images.
"""
import numpy as np
from glob import glob
from astropy.io import fits
import time,os


def subtract_overscan(frames):
    '''
    Subtracts overscan from a list images and converts them to "float32" dtype.

    Parameters
    ----------
    frames : `list`
        List of strings containing the filenames of the fits images.

    Returns
    -------
    None. Writes the overscan subtracted images as "o" + original filename to
    the disk.
    '''
    for frame in frames:
        start_time = time.time()
        hdul = fits.open(frame, output_verify = 'ignore')
        for ext in hdul[1:]:
            chip_data = ext.data.astype('float32')
            for i in range(len(chip_data)):
                row = chip_data[i]
                overscan = np.median(row[-48:]) # should I use prescan as well?
                chip_data[i] = row - overscan
            ext.data = chip_data
        filename = 'o'+frame
        hdul.writeto(filename, output_verify = 'ignore', overwrite = True)
        hdul.close()
        print(filename+" created in %s seconds" % (time.time() - start_time))

def mk_master_bias(bias_frames, debug = False):
    '''
    DEPRECATED!!! use mk_master_bias_GNU!

    Makes a master bias image by combining a list of bias images taken during
    the same night.

    Parameters
    ----------
    bias_frames : `list`
        List of strings containing the filenames of the fits bias images.

    Returns
    -------
    None. Writes the master bias to disk as BIAS_*date*.fits.
    '''
    if debug == True:
        start_time = time.time()
    
    hduls = [fits.open(frame, output_verify = 'ignore') for frame in bias_frames]
    chips = [ext.header['EXTNAME'] for ext in hduls[0][1:]]

    for chip in chips:
        data = [hdul[chip].data for hdul in hduls]
        data_array = np.asarray(data)
        data_master = np.median(data_array, axis=0)
        hduls[0][chip].data = data_master

    filename = 'BIAS_'+bias_frames[0][:16]+'.fits'
    hduls[0].writeto(filename, output_verify = 'ignore', overwrite = True)    

    for hdul in hduls: hdul.close()

    if debug == True:
        print("--- %s seconds ---" % (time.time() - start_time))

def mk_master_bias_GNU(bias_frames, night, debug = False, scale = False):
    '''
    Makes a master bias image by median combining a list of bias images taken 
    during the same night. Uses astarithmetic.

    Parameters
    ----------
    bias_frames : `list`
        List of strings containing the filenames of the fits bias images.
    night : `string`
        The night during which the images were taken

    Returns
    -------
    None. Writes the master bias to disk as BIAS_GNU_*night*.fits.
    '''
    if debug == True:
        start_time = time.time()

    hdul0 = fits.open(bias_frames[0], output_verify = 'ignore')
    chips = [ext.header['EXTNAME'] for ext in hdul0[1:]]

    if scale:
        frame_means = []
        for frame in bias_frames:
            chip_means = []
            for chip in chips:
                data = fits.getdata(frame, chip)
                chip_means.append(np.nanmean(data))
            
            frame_means.append(np.nanmean(chip_medians))

        mean_mean = np.nanmean(frame_means)

        for i in range(len(bias_frames)):
            for chip in chips:
                data = fits.getdata(bias_frames[i], chip)
                data *= mean_mean / frame_means[i]
                hdul0[chip].data = data
            filename = 'temp_' + bias_frames[i]
            hdul0.writeto(filename, output_verify = 'ignore', overwrite = True)

        temp_frames = 'temp_' + np.char.array(bias_frames)
    else:
        temp_frames = bias_frames
        
    for chip in chips:
        cmd = 'astarithmetic %s %s %s %s %s %s %s %s %s %s 10 3 0.2 sigclip-median --out=' % tuple(temp_frames) + chip + '.fits --globalhdu=' + chip
        os.system(cmd)
        hdul0[chip].data = fits.getdata(chip + '.fits')
        os.system('rm ' + chip + '.fits')

    filename = 'BIAS_GNU_' + night + '.fits'
    hdul0.writeto(filename, output_verify = 'ignore', overwrite = True)

    if scale:
        os.system('rm temp_*fits')
    hdul0.close()

    if debug == True:
        print("--- %s seconds ---" % (time.time() - start_time))

def subtract_bias(frames, master, debug = False):
    '''
    Subtracts a master bias image by from a list of images. Uses astarithmetic.

    Parameters
    ----------
    frames : `list`
        List of strings containing the filenames of the fits images to be bias
        subtracted.
    master : `string`
        The filename of the master bias image.

    Returns
    -------
    None. Writes the bias subtracted images to the disk as "z" + filename. 
    '''
    if debug == True:
        start_time = time.time()

    if len(frames) == 0: return

    for frame in frames:
        hdul = fits.open(frame, output_verify = 'ignore')
        chips = [ext.header['EXTNAME'] for ext in hdul[1:]]
        for chip in chips:
            cmd = 'astarithmetic ' + frame + ' ' + master + ' - --globalhdu=' + chip + ' --out=' + chip + '.fits'
            os.system(cmd)
            hdul[chip].data = fits.getdata(chip + '.fits')
            os.system('rm ' + chip + '.fits')

        filename = 'z' + frame
        hdul.writeto(filename, output_verify = 'ignore', overwrite = True)
        hdul.close()
        if debug == True:
            print(filename+" --- %s seconds" % (time.time() - start_time))

def trim_overscan(frames):
    '''
    Trims the overscan are from a list of images.

    Parameters
    ----------
    frames : `list`
        List of strings containing the filenames of the fits images.

    Returns
    -------
    None. Writes the trimmed images to the disk as "t" + filename. 
    '''
    for frame in frames:
        start_time = time.time()
        hdul = fits.open(frame, output_verify = 'ignore')
        for ext in hdul[1:]:
            extname = ext.header['EXTNAME']
            extnum = int(extname[-2:])
            if extnum > 80:
                chip_data = ext.data[100:,48:-48]
                ext.header['CRPIX1'] -= 48
                ext.header['CRPIX2'] -= 100
            else:
                chip_data = ext.data[0:-100,48:-48]
                ext.header['CRPIX1'] -= 48
            ext.data = chip_data
        filename = 't'+frame
        hdul.writeto(filename, output_verify = 'ignore', overwrite = True)
        hdul.close()
        print(filename+" created in %s seconds" % (time.time() - start_time))

if __name__ == '__main__':
    start_time = time.time()

    all_files = glob('OMEGA*fits')
    if os.path.exists('o'+all_files[0]):
        print('--- Skipping overscan subtraction ---')
    else:
        print('--- Subtracting overscan ---')
        subtract_overscan(all_files)

    print('--- Subtracting bias ---')
    nights = list(dict.fromkeys([file[6:16] for file in all_files]))
    for night in nights:
        frames = np.array(glob('oOMEGA.%s*fits' % night))
        hdrs = [fits.getheader(frame) for frame in frames]
        DPR_type = np.array([hdr['HIERARCH ESO DPR TYPE'] for hdr in hdrs])
        bias_frames = frames[DPR_type == 'BIAS']
        nb_frames = frames[DPR_type != 'BIAS']

        print('--- Creating master bias for ' + night + ' ---')
        mk_master_bias_GNU(bias_frames, night)

        print('--- Subtracting bias for ' + night + ' ---')
        subtract_bias(nb_frames, 'BIAS_GNU_' + night + '.fits')

    zo_files = glob('zoOMEGA*fits')
    print('--- Trimming overscan ---')
    trim_overscan(zo_files)

    print('--- Removing temporary files ---')
    os.system('rm oOMEGA*fits')
    os.system('rm zoOMEGA*fits')

    print('--- overall runtime: %s seconds' % (time.time() - start_time))
    
