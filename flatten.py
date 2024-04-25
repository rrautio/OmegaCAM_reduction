#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Program for flat-fielding debiased OMEGACAM fits images.
"""
import numpy as np
from glob import glob
from astropy.io import fits, ascii
import time,os
from scipy import fftpack, stats, ndimage
from PIL import Image, ImageDraw
from pyraf import iraf
from astropy.wcs import WCS
    
def mk_master_flat(twilight_frames, dome_frames, ide, skyflat = None, bpm = None, vmap = None, bcm = 'BCM.fits'):
    '''
    Makes a master flat image by combining a list of twilight and dome flats.
    First median combines twilight and dome flats to create a master twilight
    flat and master dome flat. Then multiplies them with each other, taking
    high-frequency spatial Fourier modes from the dome flat and low-frequency
    spatial Fourier modes form the twilight flat.

    Parameters
    ----------
    twilight_frames : `list`
        List of strings containing the filenames of the fits twilight flats, or
        the images from which low-frequency modes are taken.
    dome_frames : `list`
        List of strings containing the filenames of the fits dome flats, or the
        images from which low-frequency modes are taken.
    ide : `string`
        Identifying string attached to the output master flat filename.
    skyflat : `string`
        Filename of an image to be used as the low-frequency master flat.
        twilight_frames are not used if skyflat is provided
    bpm : `string`
        Filename of a bad pixel map.
    vmap : `string`
        Filename of a vignetting map.

    Returns
    -------
    None. Writes the master flat to disk as FLAT_*ide*.fits, the twilight
    master as TWILIGHT_*ide*.fits, and the dome master as DOME_*ide*.fits.
    '''

    if not skyflat:
        twilight_hduls = [fits.open(frame, output_verify = 'ignore') for frame in twilight_frames]
    dome_hduls = [fits.open(frame, output_verify = 'ignore') for frame in dome_frames]

    chips = [ext.header['EXTNAME'] for ext in dome_hduls[0][1:]]

    master = []
    mn = []
    for chip in chips:
        dome = [hdul[chip].data for hdul in dome_hduls]
        dome_array = np.asarray(dome)
        dome_master = np.median(dome_array, axis=0)

        low_pass_dome = get_low_spatial_freq(dome_master)
        high_pass = dome_master / low_pass_dome
        
        if skyflat:
            twil_master = fits.getdata(skyflat, chip)
        else:
            twil = [hdul[chip].data for hdul in twilight_hduls]
            twil_array = np.asarray(twil)
            twil_master = np.median(twil_array, axis=0)

        if bpm:
            badpix = fits.getdata(bpm, chip)
            twil_master[badpix == 1] = np.NaN
        if vmap:
            vignetted = fits.getdata(vmap, chip)
            twil_master[vignetted == 1] = np.NaN
        if bcm:
            badccd = fits.getdata(bcm, chip)
            twil_master[badccd == 1] = np.NaN
        twil_master[np.isnan(twil_master)] = dome_master[np.isnan(twil_master)] / np.nanmedian(dome_master)

        low_pass = get_low_spatial_freq(twil_master.astype('float'))

        master.append(low_pass * high_pass)
        mn.append(np.nanmean(master))

        dome_hduls[1][chip].data = dome_master
        dome_hduls[2][chip].data = twil_master

    medmed = np.nanmedian(mn)
        
    for i in range(len(chips)):
        master_norm = master[i] / medmed
        dome_hduls[0][chips[i]].data = master_norm
        
    filename = 'FLAT_'+ide+'.fits'
    dome_hduls[0].writeto(filename, output_verify = 'ignore', overwrite = True)
    filename = 'DOME_'+ide+'.fits'
    dome_hduls[1].writeto(filename, output_verify = 'ignore', overwrite = True)
    filename = 'TWILIGHT_'+ide+'.fits'
    dome_hduls[2].writeto(filename, output_verify = 'ignore', overwrite = True)

    if not skyflat:
        for hdul in twilight_hduls: hdul.close()
    for hdul in dome_hduls: hdul.close()
        

def get_low_spatial_freq(ima, frac = 6., debug = False):
    '''
    Extracts the low spatial frequency Fourier modes from an image.

    Parameters
    ----------
    ima : `array`
        Image array.
    frac : `float`
        Spatial frequency fraction.

    Returns
    -------
    ifft2 : `array`
        Image array containing the low spatial frequency Fourier modes.
    '''
    #fft of image
    fft1 = fftpack.fftshift(fftpack.fft2(ima))

    #Create a low pass filter image
    x,y = ima.shape[0],ima.shape[1]
    #size of circle
    e_x,e_y=x/frac,y/frac
    #create a box 
    bbox=((x/2)-(e_x/2),(y/2)-(e_y/2),(x/2)+(e_x/2),(y/2)+(e_y/2))

    low_pass=Image.new("L",(ima.shape[0],ima.shape[1]),color=0)

    draw1=ImageDraw.Draw(low_pass)
    draw1.ellipse(bbox, fill=1)

    low_pass_np=np.array(low_pass)

    #multiply both the images
    filtered=np.multiply(fft1,low_pass_np.T)

    #inverse fft
    ifft2 = np.real(fftpack.ifft2(fftpack.ifftshift(filtered)))

    if debug:
        return ifft2, fft1, filtered
    else:
        return ifft2

def flatten(frames, master, prefix = 'f'):
    '''
    Divides a list of images with a master flat-field image. Uses astarithmetic.

    Parameters
    ----------
    frames : `list`
        List of strings containing the filenames of the fits images to be
        flat-fielded.
    master : `string`
        The filename of the master flat image.
    prefix : `string`
        String to be prepended to the output filenames.

    Returns
    -------
    None. Writes the flattened images to the disk as prefix + filename. 
    '''
    for frame in frames:
        hdul = fits.open(frame, output_verify = 'ignore')
        chips = [ext.header['EXTNAME'] for ext in hdul[1:]]
        for chip in chips:
            cmd = 'astarithmetic ' + frame + ' ' + master + ' / --globalhdu=' + chip + ' --out=' + chip + '.fits'
            os.system(cmd)
            hdul[chip].data = fits.getdata(chip + '.fits')
            os.system('rm ' + chip + '.fits')

        filename = prefix + frame
        hdul.writeto(filename, output_verify = 'ignore', overwrite = True)
        hdul.close()

def mk_masks(frames, commands = '--outliersigma=10 --detgrowquant=0.75 --qthresh=0.2 --interpnumngb=4', tsize = 150, cut_pf=0, prefix='mask_'):
    '''
    Creates a mask file by running GNUAstro package NoiseChisel on the
    image provided.
    Requires:
      - Filename of image to be masked (requires .fits extension)
      - String of additional astnoisechisel commands (optional)

    Requires that NoiseChisel is installed.
    See installation instructions on the GNU webpage:
      www.gnu.org/software/gnuastro 

    NOTE: assumes you are using flattened images (either ftz*.fits or 
    pftz*.fits)
    '''
    for frame in frames:
        hdul = fits.open(frame, output_verify = 'ignore')
        chips = [ext.header['EXTNAME'] for ext in hdul[1:]]
        for chip in chips:
            tilesize = str(tsize) + ',' + str(tsize) + ' '
            cmd = 'astnoisechisel ' + frame + ' --hdu=' + chip + ' --rawoutput --output=' + chip + '.fits --tilesize='+ tilesize + commands 
            os.system(cmd)
            try:
                hdul[chip].data = fits.getdata(chip + '.fits')
            except IOError:
                tilesize = str(tsize-25) + ',' + str(tsize-25) + ' '
                print('NoiseChisel crashed. Trying tilesize = ' + tilesize)
                cmd = 'astnoisechisel ' + frame + ' --hdu=' + chip + ' --rawoutput --output=' + chip + '.fits --tilesize='+ tilesize + commands
                os.system(cmd)
                try:
                    hdul[chip].data = fits.getdata(chip + '.fits')
                except IOError:
                    tilesize = str(tsize-50) + ',' + str(tsize-50) + ' '
                    print('NoiseChisel crashed. Trying tilesize = ' + tilesize)
                    cmd = 'astnoisechisel ' + frame + ' --hdu=' + chip + ' --rawoutput --output=' + chip + '.fits --tilesize='+ tilesize + commands
                    os.system(cmd)
                    try:
                        hdul[chip].data = fits.getdata(chip + '.fits')
                    except:
                        tilesize = str(tsize-75) + ',' + str(tsize-75) + ' '
                        print('NoiseChisel crashed. Trying tilesize = ' + tilesize)
                        cmd = 'astnoisechisel ' + frame + ' --hdu=' + chip + ' --rawoutput --output=' + chip + '.fits --tilesize='+ tilesize + commands
                        os.system(cmd)
                        hdul[chip].data = fits.getdata(chip + '.fits')
                
            os.system('rm ' + chip + '.fits')

        filename = prefix + frame[int(cut_pf):]
        hdul.writeto(filename, output_verify = 'ignore', overwrite = True)
        hdul.close()
        print('Wrote binary mask file to disk as ' + filename)

def normalize(frames, mask_frames, bpm = None, vmap = None, bcm = 'BCM.fits', subtract = False, prefix = 'n'):
    '''
    Normalizes CCDs to image level in a list of images.

    Parameters
    ----------
    frames : `list`
        List of strings containing the filenames of the fits images to be
        normalized.
    mask_frames : `string`
        List of string containing the masks.
    bpm : `string`
        Filename of a bad pixel map.
    vmap : `string`
        Filename of a vignetting map.
    prefix : `string`
        String to be prepended to the output filenames.

    Returns
    -------
    None. Writes the normalized images to the disk as prefix + filename. 
    '''
    for i in range(len(frames)):
        hdul_ima = fits.open(frames[i], output_verify = 'ignore')
        hdul_mask = fits.open(mask_frames[i], output_verify = 'ignore')
        chips = [ext.header['EXTNAME'] for ext in hdul_ima[1:]]
        chip_medians = []
        for chip in chips:
            masked = hdul_ima[chip].data  * 1.
            masked[hdul_mask[chip].data == 1] = np.NaN
            if bpm:
                badpix = fits.getdata(bpm, chip)
                masked[badpix == 1] = np.NaN
            if vmap:
                vignetted = fits.getdata(vmap, chip)
                masked[vignetted == 1] = np.NaN
            if bcm:
                badccd = fits.getdata(bcm, chip)
                masked[badccd == 1] = np.NaN
            chip_medians.append(np.nanmedian(masked))
            
        ima_median = np.nanmedian(chip_medians)

        for j in range(len(chips)):
            if subtract:
                hdul_ima[chips[j]].data -= ima_median
            else:
                hdul_ima[chips[j]].data *= ima_median / chip_medians[j]

        filename = prefix + frames[i]
        hdul_ima.writeto(filename, output_verify = 'ignore', overwrite = True)
        hdul_ima.close()
        hdul_mask.close()
        print('Wrote normalized image to disk as ' + filename)

def subtract_sky_old(frames, mask_frames, bpm = None, vmap = None, bcm = 'BCM.fits', mode = False):
    '''
    DEPRECATED
    '''
    for i in range(len(frames)):
        hdul_ima = fits.open(frames[i], output_verify = 'ignore')
        hdul_mask = fits.open(mask_frames[i], output_verify = 'ignore')
        chips = [ext.header['EXTNAME'] for ext in hdul_ima[1:]]
        for chip in chips:
            masked = hdul_ima[chip].data  * 1.
            masked[hdul_mask[chip].data == 1] = np.NaN
            if bpm:
                badpix = fits.getdata(bpm, chip)
                masked[badpix == 1] = np.NaN
            if vmap:
                vignetted = fits.getdata(vmap, chip)
                masked[vignetted == 1] = np.NaN
            if bcm:
                badccd = fits.getdata(bcm, chip)
                masked[badccd == 1] = np.NaN
                
            if mode:
                if chip == 'ESO_CCD_#77':
                    chip_m = 1
                else:
                    hdu_masked = fits.PrimaryHDU(masked)
                    hdul_temp = fits.HDUList([hdu_masked])
                    hdul_temp.writeto('temp.fits', output_verify = 'ignore', overwrite = True)
                    cmd = 'aststatistics -mode -h0 temp.fits > temp.txt'
                    os.system(cmd)
                    with open('temp.txt') as f:
                        chip_m = float(f.read()[:-1])
            else:
                chip_m = np.nanmedian(masked)
            
            hdul_ima[chip].data -= chip_m

        filename = 's' + frames[i]
        hdul_ima.writeto(filename, output_verify = 'ignore', overwrite = True)
        hdul_ima.close()
        hdul_mask.close()
        print('Wrote deskied image to disk as ' + filename)

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
            

def mask(frames, mask_frames=None, bpm=None, vmap=None, debug_chip=False, cut_pf=0, prefix='masked_'):
    '''
    Sets masked pixels to NaN in a list of images.

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
    prefix : `string`
        String to be prepended to the output filenames.

    Returns
    -------
    None. Writes the sky-subtracted images to the disk as prefix + frame.
    '''
    hdul0 = fits.open(frames[0], output_verify = 'ignore', memmap = False)
    chips = [ext.header['EXTNAME'] for ext in hdul0[1:]]
    if debug_chip:
        chips = [debug_chip]
    for i in range(len(frames)):
        for chip in chips:
            ima = fits.getdata(frames[i], chip)
            mask = fits.getdata(mask_frames[i], chip)
            ima[mask == 1] = np.NaN
            if bpm:
                badpix = fits.getdata(bpm, chip)
                ima[badpix == 1] = np.NaN
            if vmap:
                vignetted = fits.getdata(vmap, chip)
                ima[vignetted == 1] = np.NaN
            hdul0[chip].data = ima


        filename = prefix + frames[i][int(cut_pf):]
        hdul0.writeto(filename, output_verify = 'ignore', overwrite = True)
        print('Wrote masked image to disk as ' + filename)

    hdul0.close()
        
def mk_skyflat(frames, ide, debug_chip = False):
    '''
    Makes a master night-sky-flat image by combining a list of masked and 
    normalized science images.

    Parameters
    ----------
    frames : `list`
        List of strings containing the filenames of the masked and normalized
        chience images.
    ide : `string`
        Identifying string attached to the output master flat filename.

    Returns
    -------
    None. Writes the master flat to disk as SKYFLAT_*ide*.fits.
    '''
    hdul0 = fits.open(frames[0], output_verify = 'ignore', memmap = False)
    chips = [ext.header['EXTNAME'] for ext in hdul0[1:]]
    if debug_chip:
        chips = [debug_chip]
    frame_medians = []
    for i in range(len(frames)):
        chip_medians = []
        for chip in chips:
            data = fits.getdata(frames[i], chip)
            chip_medians.append(np.nanmedian(data))
            
        frame_medians.append(np.nanmedian(chip_medians))

    median_median = np.nanmedian(frame_medians)

    for i in range(len(frames)):
        for chip in chips:
            data = fits.getdata(frames[i], chip)
            data *= median_median / frame_medians[i]
            hdul0[chip].data = data
        filename = 'temp_' + frames[i]
        hdul0.writeto(filename, output_verify = 'ignore', overwrite = True)

    for chip in chips:
        num = str(len(frames))
        cmd = 'astarithmetic temp_masked_*fits ' + num + ' 3 0.2 sigclip-median --out=' + chip + '.fits --globalhdu=' + chip
        os.system(cmd)

        data = fits.getdata(chip + '.fits')
        data_norm = data / median_median
        
        hdul0[chip].data = data_norm
        os.system('rm ' + chip + '.fits')

    filename = 'SKYFLAT_' + ide + '.fits'
    if debug_chip:
        filename = 'DEBUG_' + debug_chip + filename
    hdul0.writeto(filename, output_verify = 'ignore', overwrite = True)

    hdul0.close()
    os.system('rm temp_masked_*fits')
    
def mk_skyflat_pyraf(frames):
    '''
    DEPRECATED
    Creates a flat using imcombine.  Rejects masked pixels by creating
    .pl versions of the masks and adding these to the header of the
    input images.
    Requires:
     - List of images that go into making the flat (tz*.fits)
     - List of masks associated with the above images (M*.fits)
     - Index of current loop in final flat-making process (int)
     - Name of the vignetting mask (e.g., vmap_on.fits)
     - Flag indicated whether to use the raw images (tz*) or
     plane-corrected images (ptz*)

    Set pflag to 0 for first iteration (before any planes are
    measured).
    Currently this uses ALL exposures, object and sky.
    '''
    for i in range(len(frames)):
        iraf.imcopy(input='CCDTEST_mask_f'+frames[i][8:],
                  output='CCDTEST_mask_f'+frames[i][8:]+'.pl')
        iraf.hedit(images=frames[i],
                   fields='BPM',
                   value='CCDTEST_mask_f'+frames[i][8:]+'.pl',
                   add='yes',
                   verify='no')

    instr = 'CCDTEST_tz*.fits'
    outstr = 'PYRAFFLAT_ha.fits'
    iraf.unlearn('imcombine')
    iraf.imcombine(input=instr,
                   output=outstr,
                   combine='median',
                   reject='avsigclip',
                   masktype='goodvalue',
                   maskvalue=0,
                   scale='median',
                   lthreshold=-50, # Should take care of the masking....
                   hthreshold='INDEF',
                   mclip='yes',
                   lsigma=3.0,
                   hsigma=3.0)

def erode_masks(mask_frames, iters = 5, struct = np.array([[0,1,0],[1,1,1],[0,1,0]]), debug_chip = False):
    for frame in mask_frames:
        hdul = fits.open(frame, output_verify = 'ignore')
        chips = [ext.header['EXTNAME'] for ext in hdul[1:]]
        if debug_chip:
            chips = [debug_chip]
        for chip in chips:
            mask = hdul[chip].data
            eroded = ndimage.binary_erosion(mask, struct, iters)
            hdul[chip].data = eroded.astype('int')

        filename = 'e'+frame
        if debug_chip:
            filename = 'DEBUG_' + debug_chip + filename
        hdul.writeto(filename, output_verify = 'ignore', overwrite = True)
        print('Wrote eroded mask file to disk as ' + filename)

def dilate_masks(mask_frames, iters = 5, struct = np.array([[0,1,0],[1,1,1],[0,1,0]]), debug_chip = False):
    for frame in mask_frames:
        hdul = fits.open(frame, output_verify = 'ignore')
        chips = [ext.header['EXTNAME'] for ext in hdul[1:]]
        if debug_chip:
            chips = [debug_chip]
        for chip in chips:
            mask = hdul[chip].data
            dilated = ndimage.binary_dilation(mask, struct, iters)
            hdul[chip].data = dilated.astype('int')

        filename = 'd'+frame
        if debug_chip:
            filename = 'DEBUG_' + debug_chip + filename
        hdul.writeto(filename, output_verify = 'ignore', overwrite = True)
        print('Wrote dilated mask file to disk as ' + filename)
    

if __name__ == '__main__':
    start_time = time.time()

    tzo_files = np.array(glob('tzoOMEGA*fits'))
    hdrs = [fits.getheader(frame, output_verify = 'ignore') for frame in tzo_files]
    twil_ha_frames = []
    dome_ha_frames = []
    sci_ha_frames = []
    twil_r_frames = []
    dome_r_frames = []
    sci_r_frames = []
    for i in range(len(hdrs)):
        if hdrs[i]['HIERARCH ESO INS FILT1 NAME'] == 'NB_659' and hdrs[i]['HIERARCH ESO DPR TYPE'] == 'FLAT,SKY':
            twil_ha_frames.append(tzo_files[i])
        elif hdrs[i]['HIERARCH ESO INS FILT1 NAME'] == 'NB_659' and hdrs[i]['HIERARCH ESO DPR TYPE'] == 'FLAT,DOME':
            dome_ha_frames.append(tzo_files[i])
        elif hdrs[i]['HIERARCH ESO INS FILT1 NAME'] == 'r_SDSS' and hdrs[i]['HIERARCH ESO DPR TYPE'] == 'FLAT,SKY':
            twil_r_frames.append(tzo_files[i])
        elif hdrs[i]['HIERARCH ESO INS FILT1 NAME'] == 'r_SDSS' and hdrs[i]['HIERARCH ESO DPR TYPE'] == 'FLAT,DOME':
            dome_r_frames.append(tzo_files[i])
        elif hdrs[i]['HIERARCH ESO INS FILT1 NAME'] == 'NB_659' and hdrs[i]['HIERARCH ESO DPR TYPE'] == 'OBJECT':
            sci_ha_frames.append(tzo_files[i])
        elif hdrs[i]['HIERARCH ESO INS FILT1 NAME'] == 'r_SDSS' and hdrs[i]['HIERARCH ESO DPR TYPE'] == 'OBJECT':
            sci_r_frames.append(tzo_files[i])
            
    if os.path.exists('FLAT_Ha.fits'):
        print('--- Skipping Halpha flat creation ---')
    else:
        print('--- Creating Halpha master flat ---')
        mk_master_flat(twil_ha_frames, dome_ha_frames, 'Ha')

    print('--- Flattening Halpha science images ---')
    flatten(sci_ha_frames, 'FLAT_Ha.fits')

    if os.path.exists('FLAT_r.fits'):
        print('--- Skipping r-band flat creation ---')
    else:
        print('--- Creating r-band master flat ---')
        mk_master_flat(twil_r_frames, dome_r_frames, 'r')

    print('--- Flattening r-band science images ---')
    flatten(sci_r_frames, 'FLAT_r.fits')

    f_sci_r = 'f' + np.char.array(sci_r_frames)
    f_sci_ha = 'f' + np.char.array(sci_ha_frames)

    r_masks = 'mask_f' + np.char.array(sci_r_frames)
    ha_masks = 'mask_f' + np.char.array(sci_ha_frames)
    
    if os.path.exists(r_masks[0]):
        print('--- Skipping mask creation and using old masks ---')
    else:
        print('--- Making masks (Halpha) ---')
        mk_masks(f_sci_ha)
    
        print('--- Making masks (r-band) ---')
        mk_masks(f_sci_r)


    print('--- Normalizing all debiased CCDs to image level (Halpha) ---')
    normalize(sci_ha_frames, ha_masks, bpm = 'BPM_ha.fits', vmap = 'ha_cross.fits', bcm = 'BCM.fits')

    print('--- Normalizing all debiased CCDs to image level (r-band) ---')
    normalize(sci_r_frames, r_masks, bpm = 'BPM_r.fits', bcm = 'BCM.fits')

    n_sci_r = 'n' + np.char.array(sci_r_frames)
    n_sci_ha = 'n' + np.char.array(sci_ha_frames)

    print('--- Applying masks to normalized images (Halpha) ---')

    mask(n_sci_ha, ha_masks)

    print('--- Applying masks to normalized images (r-band) ---')

    mask(n_sci_r, r_masks)

    masked_r = 'masked_n' + np.char.array(sci_r_frames)
    masked_ha = 'masked_n' + np.char.array(sci_ha_frames)

    print('--- Making Halpha skyflat from masked and normalized science images ---')
    mk_skyflat(masked_ha, 'Ha')

    print('--- Making r-band skyflat from masked and normalized science images ---')
    mk_skyflat(masked_r, 'r')

    print('--- Making a Halpha master flat from skyflat and domeflats ---')
    mk_master_flat([], dome_ha_frames, 'sky_Ha', skyflat = 'SKYFLAT_Ha.fits', bpm = 'BPM_ha.fits', vmap = 'ha_cross.fits', bcm = 'BCM.fits')

    print('--- Making r-band master flat from skyflat and domeflats ---')
    mk_master_flat([], dome_r_frames, 'sky_r', skyflat = 'SKYFLAT_r.fits', bpm = 'BPM_r.fits', bcm = 'BCM.fits')

    print('--- Final flattening (Halpha) ---')
    flatten(sci_ha_frames, 'FLAT_sky_Ha.fits', prefix = 'F')

    print('--- Final flattening (r-band) ---')
    flatten(sci_r_frames, 'FLAT_sky_r.fits', prefix = 'F')

    F_sci_r = 'F' + np.char.array(sci_r_frames)
    F_sci_ha = 'F' + np.char.array(sci_ha_frames)

    print('--- Subtractin sky (Halpha) ---')
    subtract_sky(F_sci_ha, ha_masks, bpm = 'BPM_ha.fits', vmap = 'ha_cross_leak.fits')

    print('--- Subtractin sky (r-band) ---')
    subtract_sky(F_sci_r, r_masks, bpm = 'BPM_r.fits')

    print('--- Removing temporary files ---')
    os.system('rm ftzoOMEGA*fits')
    os.system('rm ntzoOMEGA*fits')
    os.system('rn masked_ntzoOMEGA*fits')


    print('--- overall runtime: %s seconds' % (time.time() - start_time))

