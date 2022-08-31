# OmegaCAM_reduction
OmegaCAM M83 Halpha data reduction software

Dependencies:
  - GNUastro
  - SCAMP
  - SWarp
  
Reduction steps:
  1. run debias.py in the directory containing the OMEGA*fits files 
  2. run flatten.py
  3. run coadd.py
  4. run build_psf.py
  5. build the outer parts of the PSF manually
  6. run star_subtraction.py
  
