import os
import numpy as np
from astropy.io import fits
from glob import glob
import subprocess
from scipy.optimize import curve_fit

def mk_kernel():
    '''
    Creates a kernel for convolution.
    '''
    cmd = 'astmkprof --kernel=gaussian,2,5 --oversample=1 -okernel.fits'
    os.system(cmd)

def segment(frame, out, saturation = 60000, nc_commands = '--interpnumngb=100 --detgrowquant=0.8 --detgrowmaxholesize=100000', seg_commands = '--gthresh=-10 --objbordersn=0'):
    '''
    Creates a segmentation map for stamp creation.
    '''
    # mask saturated pixels as NaN
    cmd = 'astarithmetic ' + frame + ' set-i i i ' + str(saturation) + \
          ' gt 2 dilate 2 dilate 2 dilate 2 dilate nan where ' \
          '--output=tmp_masked_sat.fits -h0'
    os.system(cmd)
        
    # convolve the image for NoiseChisel and Segment
    cmd = 'astconvolve tmp_masked_sat.fits --kernel=kernel.fits ' + \
            '--domain=spatial --output=tmp_masked_conv.fits'
    os.system(cmd)

    # fill the masks
    cmd = 'astarithmetic tmp_masked_sat.fits 2 interpolate-maxofregion ' + \
          '--output=tmp_fill.fits'
    os.system(cmd)
    cmd = 'astarithmetic tmp_masked_conv.fits 2 interpolate-maxofregion' + \
          ' --output=tmp_fill_conv.fits'
    os.system(cmd)
    os.system('rm tmp_masked_conv.fits')

    # NoiseChisel
    cmd = 'astnoisechisel tmp_fill.fits ' + nc_commands + \
            ' --convolved=tmp_fill_conv.fits --output=tmp_nc.fits'
    os.system(cmd)
    os.system('rm tmp_fill.fits')

    # Segment
    cmd = 'astsegment tmp_nc.fits --output=tmp_seg.fits' + \
            ' --convolved=tmp_fill_conv.fits --rawoutput ' + \
            seg_commands
    os.system(cmd)
    
    hdul_masked = fits.open('tmp_masked_sat.fits')
    hdul_seg = fits.open('tmp_seg.fits')
    hdul_masked.append(hdul_seg['CLUMPS'])
    hdul_masked.append(hdul_seg['OBJECTS'])
    hdul_masked.writeto(out, overwrite=True)
    
    os.system('rm tmp_fill_conv.fits tmp_nc.fits tmp_masked_sat.fits tmp_seg.fits')

def mk_stamps(segment, normradii = (5,10), stampwidth = 500,  magrange = (12,13), mindist=0.01, stampdir = 'tmp_stamps'):
    '''
    Creates scaled and masked stamp images of stars in a coadd.
    '''
    os.system('mkdir ' + stampdir)
    os.system('rm -r ' + stampdir + '/*')
    # Make a catalog of stars
    cmd = 'astscript-psf-select-stars ' + segment + \
          ' --magnituderange=' + str(magrange[0]) + ',' + str(magrange[1]) + \
          ' --mindistdeg=' + str(mindist) + ' --output=tmp_starcat.fits'
    os.system(cmd)
        
    # Make the stamps
    tab = fits.open('tmp_starcat.fits')
    i = 0
    for row in tab[1].data:
        cmd = 'astscript-psf-stamp ' + segment + ' --mode=wcs ' + \
                ' --stampwidth=' + str(stampwidth) + ',' + str(stampwidth) + \
                ' --center=' + str(row['ra']) + ',' + str(row['dec']) + \
                ' --normradii=' + \
                str(normradii[0]) + ',' + str(normradii[1]) + \
                ' --segment=' + segment + \
                ' --output=' + stampdir + '/' + str(i) + '.fits'
        os.system(cmd)
        i += 1

    os.system('rm tmp_starcat.fits')

def mask_reflections(stampdir):
    os.system('rm ' + stampdir + '/*rs.fits')
    stamps = glob(stampdir + '/*fits')
    for stamp in stamps:
        cmd = 'astnoisechisel --outliersigma=10 --tilesize=30,30 ' + \
            '--meanmedqdiff=0.1 --output=tmp_strong.fits ' + \
            stamp
        os.system(cmd)
        cmd = 'astnoisechisel --detgrowmaxholesize=150 ' + \
            ' --interpnumngb=4 --meanmedqdiff=0.1 ' + \
            '--output=tmp_weak.fits ' + stamp
        os.system(cmd)
        cmd = 'astarithmetic tmp_strong.fits -hDETECTIONS tmp_weak.fits ' + \
              '-hDETECTIONS 0 where 1 connected-components ' + \
              '--output=tmp_ghostdet.fits'
        os.system(cmd)
        cmd = 'astmkcatalog tmp_ghostdet.fits -h1 --ids --area ' + \
              '--output=tmp_cat.fits'
        os.system(cmd)
        lab = subprocess.check_output(['asttable', 'tmp_cat.fits',
                                       '--sort=area', '--tail=1', '-c1'])[:-1]
        os.system(cmd)
        cmd = 'astarithmetic ' + stamp + ' -h1 tmp_ghostdet.fits -h1 ' + lab + \
              ' eq nan where --output=' + stamp[:-5] + '_rs.fits'
        os.system(cmd)
    os.system('rm tmp_strong.fits')
    os.system('rm tmp_weak.fits')
    os.system('rm tmp_cat.fits')
    os.system('rm tmp_ghostdet.fits')

def combine_stamps(out, stampdir = 'tmp_stamps'):
    '''
    Combines stamp images from a given directory into a PSF model.
    '''
    n = len(glob(stampdir + '/*.fits'))
    cmd = 'astarithmetic ' + stampdir + '/*.fits ' + str(n) + \
          ' 3 0.2 sigclip-mean -g1 --wcsfile=none --output=' + out
    os.system(cmd)

def unite(outer, inner, center = 501, normradii = (10,12,15), out = 'psf.fits'):
    '''
    Unites two PSF models into one.
    '''
    cmd = ['astscript-psf-scale-factor',
           outer,
           '--psf=' + inner,
           '--center=' + str(center) + ',' + str(center),
           '--mode=img',
           '--normradii=' + str(normradii[0]) + ',' + str(normradii[2]),
           '--quiet']
    scale = float(subprocess.check_output(cmd))
    cmd = 'astscript-psf-unite ' + outer + ' --inner=' + inner + \
            ' --radius=' + str(normradii[1]) + ' --scale=' + str(scale) + \
            ' --output=' + out
    os.system(cmd)
    
def mk_stamps_mosaic(frame, saturation = 60000):
    '''
    Creates stamps from multifits mosaic.
    NOT IN USE!
    '''
    hdul = fits.open(frame)
    chips = [ext.header['EXTNAME'] for ext in hdul[1:]]
    os.system('mkdir tmp_stamps')
    os.system('rm tmp_stamps/*')
    for chip in chips:
        # mask saturated pixels as NaN
        cmd = 'astarithmetic ' + frame + ' set-i i i ' + str(saturation) + \
              ' gt 2 dilate 2 dilate 2 dilate 2 dilate nan where ' \
              '--output=tmp_masked_sat.fits -h' + chip
        os.system(cmd)
        
        # convolve the image for NoiseChisel and Segment
        cmd = 'astconvolve tmp_masked_sat.fits --kernel=kernel.fits ' + \
                '--domain=spatial --output=tmp_masked_conv.fits'
        os.system(cmd)

        # fill the masks
        cmd = 'astarithmetic tmp_masked_sat.fits 2 interpolate-maxofregion ' + \
              '--output=tmp_fill.fits'
        os.system(cmd)
        cmd = 'astarithmetic tmp_masked_conv.fits 2 interpolate-maxofregion' + \
              ' --output=tmp_fill_conv.fits'
        os.system(cmd)
        os.system('rm tmp_masked_conv.fits')

        # NoiseChisel
        cmd = 'astnoisechisel tmp_fill.fits --interpnumngb=100 ' + \
                '--detgrowquant=0.8 --detgrowmaxholesize=100000 ' + \
                '--convolved=tmp_fill_conv.fits --output=tmp_nc.fits'
        os.system(cmd)

        # Segment
        cmd = 'astsegment tmp_nc.fits --output=' + \
                'tmp_seg.fits ' + \
                '--convolved=tmp_fill_conv.fits --rawoutput ' + \
                '--gthresh=-10 --objbordersn=0'
        os.system(cmd)
        os.system('rm tmp_fill_conv.fits tmp_nc.fits tmp_masked_sat.fits')

        # Make a catalog of stars
        cmd = 'astscript-psf-select-stars ' + frame + ' -h' + chip + \
              ' --magnituderange=12,13 --mindistdeg=0.01 ' + \
              '--output=tmp_12-13.fits'
        os.system(cmd)
        
        # Make the stamps
        tab = fits.open('tmp_12-13.fits')
        i = 0
        for row in tab[1].data:
            cmd = 'astscript-psf-stamp ' + frame + ' --mode=wcs ' + \
                    '-h' + chip + ' --stampwidth=500 --center=' +  \
                    str(row['ra']) + ',' + str(row['dec']) + \
                    ' --normradii=5,10 --segment=tmp_seg.fits ' + \
                    '--output=tmp_stamps/' + chip + '_' + str(i) + '.fits'
            os.system(cmd)
            i += 1

def get_star_scales(frame, psf, output = 'cat.fits', magrange=(6,10), mindist=0, normradii='auto', segment = 'same', tst_ima = 'default'):
    '''
    Subtract stars from an image.
    '''
    if segment == 'same':
        segment = frame
    # make a catalog of stars
    cmd = 'astscript-psf-select-stars ' + frame + \
          ' --magnituderange=' + str(magrange[0]) + ',' + str(magrange[1]) + \
          ' --mindistdeg=' + str(mindist) + ' --output=' + output
    os.system(cmd)
    
    cat = fits.open(output)
    tmp_file = 'tmp_subtracted.fits'
    frame0 = frame
    scale_factor = []
    for star in cat[1].data:
        if normradii == 'auto':
            nr = 20 + 10 * 10**((star['phot_g_mean_mag'] - 10)/-2.5)
            print('normradii ' + str(nr))
            nr = (int(nr), int(nr) + 5)
        else:
            nr = normradii
        center = str(star[0]) + ',' + str(star[1])
        # get the scale-factor'
        cmd = ['astscript-psf-scale-factor',
               frame,
               '--psf=' + psf,
               '--center=' + center,
               '--mode=wcs',
               '--normradii=' + str(nr[0]) + ',' + str(nr[1]),
               '--segment=' + segment,
               '--quiet']
        try:
            scale = float(subprocess.check_output(cmd))
        except:
            scale_factor.append(0.0)
            continue

        scale_factor.append(scale)

        if tst_ima:
            # subtract the star'

            cmd = 'astscript-psf-subtract ' + frame + ' --mode=wcs ' + \
                    '--psf=' + psf + ' --scale=' + str(scale) + \
                    ' --center=' + center + ' --output=tmp_out.fits'
            os.system(cmd)
            os.system('cp tmp_out.fits ' + tmp_file)
            frame = tmp_file

    newcol = fits.Column(name = 'scale_factor', format = 'D', array = np.array(scale_factor))
    newhdu = fits.BinTableHDU.from_columns(cat[1].data.columns + newcol)
    cat[1] = newhdu
    cat.writeto(output, overwrite = True)

    if tst_ima == 'default':
        os.system('cp ' + tmp_file + ' ' + frame0[:-5] + '_starsub.fits')
    elif tst_ima:
        os.system('cp ' + tmp_file + ' ' + tst_ima)
    os.system('rm tmp_subtracted.fits tmp_out.fits')

def cull_star_scales(starcat, culled_starcat):
    cat = fits.open(starcat)
    galcen = (204.253958, -29.865417)
    exr = 0.12 # deg
    for star in cat[1].data:
        if (star['ra'] - galcen[0])**2 + (star['dec'] - galcen[1])**2 < exr**2:
            star['scale_factor'] = 0.
            cat.writeto(culled_starcat, overwrite=True)

#-----------------------------------------------------------------------------
def radial_profile(filename, center=(3895,1664), pix_scale=1, max_r=20000, rbin=1):
    '''
    Calculates radial profile from a fits file, requires:
        --filename, the fits file
        --center, the pixel coordinates of the center
        --pix_scale, the pixel to arcsecond scale of the image
        --max_r, the extent of the profile
        -rbin, size of the radial bin in pixels
    Returns:
        --(flux, rr) the intensity as a function of radius 
    ''' 
    data=fits.getdata(filename)
    x, y = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)/rbin
    r = r.astype(np.int)
    rm = r.ravel()[data.ravel() > -900]
    datam = data.ravel()[data.ravel() > -900]
    nr = np.bincount(rm)
    tbin = np.bincount(rm, datam)
    radialprofile = tbin / nr
    flux = radialprofile[:max_r/rbin]
    rr = np.arange(len(flux)) * rbin * pix_scale
    return flux, rr
    
def desky_psf(psf_3d, psf_2d, out=False, cen3d=(1000,1000), cen2d=(3500,3500), normradii=(150,160)):
    f3d, r3d = radial_profile(psf_3d, cen3d)
    f2d, r2d = radial_profile(psf_2d, cen2d)
    f2d = f2d * np.median(f3d[normradii[0]:normradii[1]])/np.median(f2d[normradii[0]:normradii[1]])

    def func(r,const):
        return f3d[r] + const

    popt, pcov = curve_fit(func,r2d[normradii[0]:1414],f2d[normradii[0]:1414],p0=(0))
    skyval = popt[0]

    if out:
        cmd = 'astarithmetic ' + psf_3d + ' ' + str(skyval) + ' + --out=' + out
        os.system(cmd)

    return -1*skyval
    

if __name__ == '__main__':
    coadd_ha = 'coadd_ha.fits'
    seg_ha = 'coadd_ha_seg.fits'
    outer_ha = 'mid_ha.fits'
    inner_ha = 'inner_ha.fits'
    psf_ha = 'psf_ha_inner.fits'
    
    coadd_r = 'coadd_r.fits'
    seg_r = 'coadd_r_seg.fits'
    outer_r = 'mid_r.fits'
    inner_r = 'inner_r.fits'
    psf_r = 'psf_r_inner.fits'

    mk_kernel()

    print('--- Making segmentation map for Halpha ---')

    segment(coadd_ha, seg_ha)

    print('--- Making the outer part of the PSF for Halpha ---')

    mk_stamps(seg_ha, normradii = (10,15), stampwidth = 500,  magrange = (10,12), mindist=0.01, stampdir = 'stamps_mid')
    combine_stamps(outer_ha, stampdir = 'stamps_mid')
    os.system('rm -r stamps_mid')

    print('--- Making the inner part of the PSF for Halpha ---')

    mk_stamps(seg_ha, normradii = (5,10), stampwidth = 500,  magrange = (12,13), mindist=0.01, stampdir = 'stamps_inner')
    combine_stamps(inner_ha, stampdir = 'stamps_inner')
    os.system('rm -r stamps_inner')

    print('--- Uniting the two PSF parts for Halpha ---')

    unite(outer_ha, inner_ha, center = 251, normradii = (7,10,13), out = psf_ha)
    
    print('--- Making segmentation map for r-band ---')

    segment(coadd_r, seg_r)

    print('--- Making the outer part of the PSF for r-band ---')

    mk_stamps(seg_r, normradii = (10,15), stampwidth = 500,  magrange = (10,12), mindist=0.01, stampdir = 'stamps_mid')
    combine_stamps(outer_r, stampdir = 'stamps_mid')
    os.system('rm -r stamps_mid')

    print('--- Making the inner part of the PSF for r-band ---')

    mk_stamps(seg_r, normradii = (5,10), stampwidth = 500,  magrange = (12,13), mindist=0.01, stampdir = 'stamps_inner')
    combine_stamps(inner_r, stampdir = 'stamps_inner')
    os.system('rm -r stamps_inner')

    print('--- Uniting the two PSF parts for r-band ---')

    unite(outer_r, inner_r, center = 251, normradii = (7,10,13), out = psf_r)
