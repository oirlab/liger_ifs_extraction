import numpy as np
import os
from astropy.io import fits
import pandas as pd
from numba import njit

from contextlib import contextmanager
from time import time

####################################
#### Flux conserving resampling ####
####################################

@njit(inline="always")
def _bin_left(x, i):
    ''' Given an array of bin centers x, return the left edge of the i-th bin. '''
    if i == 0:
        return x[0] - 0.5 * (x[1] - x[0])
    return 0.5 * (x[i - 1] + x[i])


@njit(inline="always")
def _bin_right(x, i, n):
    ''' Given an array of bin centers x, return the right edge of the i-th bin. '''
    if i == n - 1:
        return x[n - 1] + 0.5 * (x[n - 1] - x[n - 2])
    return 0.5 * (x[i] + x[i + 1])


@njit(inline="always")
def _advance_to_overlap(x_in, j, xl, n_in):
    ''' Advance the index j in x_in until x_in[j] is the last point before xl. '''
    while j < n_in - 1 and x_in[j + 1] <= xl:
        j += 1
    return j


@njit(inline="always")
def _integrate_linear_segment(x0, x1, y0, y1, a, b):
    ''' Integrate a linear segment defined by points (x0, y0) and (x1, y1) over the interval [a, b]. '''
    slope = (y1 - y0) / (x1 - x0)

    da = a - x0
    db = b - x0

    return (
        y0 * (db - da)
        + 0.5 * slope * (db * db - da * da)
    )


@njit(cache=True)
def resample_flux_conserving_1d(x_in, flux_in, x_out):
    ''' Resample flux_in from x_in to x_out while conserving total flux. '''
    n_in = x_in.size
    n_out = x_out.size

    flux_out = np.zeros(n_out, dtype=flux_in.dtype)

    j = 0

    for i in range(n_out):

        xl = _bin_left(x_out, i)
        xr = _bin_right(x_out, i, n_out)

        j = _advance_to_overlap(x_in, j, xl, n_in)

        k = j
        total = 0.0

        while k < n_in - 1 and x_in[k] < xr:

            seg_l = xl if xl > x_in[k] else x_in[k]
            seg_r = xr if xr < x_in[k + 1] else x_in[k + 1]

            if seg_r > seg_l:
                total += _integrate_linear_segment(
                    x_in[k],
                    x_in[k + 1],
                    flux_in[k],
                    flux_in[k + 1],
                    seg_l,
                    seg_r,
                )

            k += 1

        flux_out[i] = total / (xr - xl)

    return flux_out

####################
#### Linear fit ####
####################

@njit(nogil=True, cache=True)
def linear_fit_1d(x, y):
    ''' Perform a linear fit to the data points (x, y) and return the slope and intercept. '''
    n = x.size

    sx = 0
    sy = 0
    sxx = 0
    sxy = 0

    for i in range(n):
        xi = x[i]
        yi = y[i]

        sx += xi
        sy += yi
        sxx += xi * xi
        sxy += xi * yi

    denom = n * sxx - sx * sx

    slope = (n * sxy - sx * sy) / denom if denom != 0 else 0
    b = (sy - slope * sx) / n if n != 0 else 0

    return slope, b

######################################
#### Convolution and PSF sampling ####
######################################

@njit
def convolve2d_numba(image, kernel):
    ny, nx = image.shape
    ky, kx = kernel.shape
    pad_y = ky // 2
    pad_x = kx // 2

    output = np.zeros_like(image)

    for i in range(pad_y, ny - pad_y):
        for j in range(pad_x, nx - pad_x):
            s = 0.0
            for m in range(ky):
                for n in range(kx):
                    s += image[i - pad_y + m, j - pad_x + n] * kernel[m, n]
            output[i, j] = s

    return output

@njit
def sample_psf_numba(epsf):
    sampled_epsf = np.zeros((15,15,3,3), dtype=np.float32)
    for i in range(15):
        for j in range(15):
            for k in range(3):
                for l in range(3):
                    sampled_epsf[i,j,k,l] = epsf[8+i+15*k, 8+j+15*l]
    return sampled_epsf

@njit
def generate_flux(
    arr_pix, x_samp, flux_samp, sampledepsf, image
):
    for j in range(len(arr_pix)):
        a = arr_pix[j]
        b = x_samp[j]
        x = a // 15
        oa = int(a % 15)
        y = b // 15
        ob = int(b % 15)
        image[y-1:y+2, x-1:x+2] += flux_samp[j] * sampledepsf[oa, ob]
    return


# Main function to create raw frame from input cube and trace data

@njit(nogil=True, cache=True)
def make_liger_ifu_image(
    output_image : np.ndarray,
    x1, x2, x3, x4, x5,
    y1, y2, y3, y4, y5,
    wave_filt : np.ndarray,
    cube : np.ndarray,
    arr_mask : np.ndarray,
    wave : np.ndarray,
    sampled_epsf : np.ndarray,
) -> np.ndarray:

    nx1 = len(x1)

    xarr = np.zeros((5,), dtype=np.float32)
    yarr = np.zeros((5,), dtype=np.float32)

    for i in range(nx1):

        if (
            (x1[i] < 0)
            or (x1[i] > 61439)
            or (x5[i] < 0)
            or (x5[i] > 61439)
            or (y1[i] < 0)
            or (y1[i] > 61439)
            or (y5[i] < 0)
            or (y5[i] > 61439)
        ):
            continue

        loc = np.where(arr_mask == i)
        flux_arr = cube[:, loc[1][0], loc[0][0]]

        xarr[0] = x1[i]
        xarr[1] = x2[i]
        xarr[2] = x3[i]
        xarr[3] = x4[i]
        xarr[4] = x5[i]

        yarr[0] = y1[i]
        yarr[1] = y2[i]
        yarr[2] = y3[i]
        yarr[3] = y4[i]
        yarr[4] = y5[i]
        
        #Polynomial fit trace
        coeff = linear_fit_1d(yarr, xarr)

        #Polynomial fit wave y values with wavesampling
        coeff_wave = linear_fit_1d(yarr, wave_filt)

        #Create pixel sampling space in y. 
        arr_pix = np.arange(y5[i], y1[i] + 1)

        #Create corresponding wave sampling
        wave_samp = coeff_wave[0] * arr_pix + coeff_wave[1]
        
        flux_samp = resample_flux_conserving_1d(wave, flux_arr, wave_samp[::-1])

        x_samp = coeff[0] * arr_pix + coeff[1]

        generate_flux(arr_pix, x_samp, flux_samp, sampled_epsf, output_image)

    return output_image


def load_filter_data(filter_name : str | None = None):
    
    #Load Filter Files
    df = pd.read_csv("data/filter.csv", delimiter=" ")

    #Loading filterlist and corresponding micropupil list
    filter_list = df["filter"].values
    micropupil_list = df["micropupil_file"].values
    min_wave_list = df["min_wave"].values
    max_wave_list = df["max_wave"].values

    #Find micropupil corresponding to the filter
    if filter_name is None:
        return df
    else:
        ind_filt = np.where(filter_name==filter_list)[0][0]
        return {
            'micropupil_file': micropupil_list[ind_filt],
            'min_wave': min_wave_list[ind_filt],
            'max_wave': max_wave_list[ind_filt],
        }


def simulate_lenslet_raw_frame(
    input_cube : np.ndarray,
    arr_mask : np.ndarray,
    micropupil_dir : str,
    trace_dir : str,
    filter_name : str = 'KN2',
    resolution : str ='4000',
    itime : float = 900*5,
    n_frames : int = 1,
    read_noise : float = 5,
    dark_current : float = 0.002
) -> dict:
    
    #Extract data
    data = np.loadtxt(trace_dir + filter_name + '_' + resolution + '.csv', delimiter=' ')
    
    x1 = data[:, 4]
    x2 = data[:, 6]
    x3 = data[:, 8]
    x4 = data[:, 10]
    x5 = data[:, 12]

    y1 = data[:, 5]
    y2 = data[:, 7]
    y3 = data[:, 9]
    y4 = data[:, 11]
    y5 = data[:, 13]

    # Load filter info
    filter_info = load_filter_data(filter_name)

    #Load micropupil data, normalize, and compute sampled effective point spread function
    mpupil = fits.getdata(micropupil_dir + filter_info['micropupil_file'])

    # Normalize and compute effective PSF
    mpupil = mpupil / mpupil.sum()
    epsf = convolve2d_numba(mpupil, np.ones((15, 15)))
    sampled_epsf = sample_psf_numba(epsf)

    # Convert to 1 micron pixel grid locations. 
    # All convolutions will be done on this 1 micron detector and then binned to get 4kx4k
    x1 = np.round((x1) / 0.001) + 2048 * 15
    x2 = np.round((x2) / 0.001) + 2048 * 15
    x3 = np.round((x3) / 0.001) + 2048 * 15
    x4 = np.round((x4) / 0.001) + 2048 * 15
    x5 = np.round((x5) / 0.001) + 2048 * 15

    y1 = np.round((y1) / 0.001) + 2048 * 15
    y2 = np.round((y2) / 0.001) + 2048 * 15
    y3 = np.round((y3) / 0.001) + 2048 * 15
    y4 = np.round((y4) / 0.001) + 2048 * 15
    y5 = np.round((y5) / 0.001) + 2048 * 15

    x1 = x1.astype(int)
    x2 = x2.astype(int)
    x3 = x3.astype(int)
    x4 = x4.astype(int)
    x5 = x5.astype(int)

    y1 = y1.astype(int)
    y2 = y2.astype(int)
    y3 = y3.astype(int)
    y4 = y4.astype(int)
    y5 = y5.astype(int)

    # Load lenslet data cube in e/s
    wave = np.linspace(filter_info['min_wave'], filter_info['max_wave'], input_cube.shape[0])
    
    # Create wave sampling at 5 points
    wave_filt = np.arange(wave.min(), wave.max() + .01, (wave.max() - wave.min()) / 4)

    # Create image
    shape = (4096, 4096)
    final_image = np.zeros(shape, dtype=np.float32)
    make_liger_ifu_image(
        final_image,
        x1, x2, x3, x4, x5,
        y1, y2, y3, y4, y5,
        wave_filt, input_cube, arr_mask, wave, sampled_epsf
    )

    # Add Dark current and read noise
    final_image_rate = final_image + dark_current
    final_image_tot = final_image_rate * itime * n_frames + read_noise**2 * n_frames
    
    # Add poisson noise
    if final_image_tot.min() > 0:
        sim_tot_noise = np.random.poisson(lam=final_image_tot, size=shape).astype(np.float32)
    else:
        sim_tot_noise = final_image_tot
    
    sim_rate_noise = sim_tot_noise / (itime * n_frames)

    return {
        'sim_perf' : final_image_rate,
        'sim' : sim_rate_noise,
        'var' : sim_tot_noise / (itime * n_frames)**2
    }

def save_simulated_lenslet_raw_frame(
    sim_data : dict,
    output_fn : str = 'simulated_lenslet_raw_frame.fits'
):
    primary_hdu = fits.PrimaryHDU()
    hdu1 = fits.ImageHDU(data=sim_data['sim'], name='IM')
    hdu2 = fits.ImageHDU(data=sim_data['var'], name='VAR')
    hdulist = fits.HDUList([primary_hdu, hdu1, hdu2])
    hdulist.writeto(output_fn, overwrite=True)


@contextmanager
def timer():
    start = time()
    yield
    end = time()
    print(f"Elapsed time: {end - start:.2f} seconds")

##############################################
####### Making recmats from lamp scan ########
##############################################

@njit
def lenslet_index_to_scan_index(i:int, arr_mask:np.ndarray):
    loc=np.where(arr_mask==i)
    ilens = loc[1][0]
    jlens = loc[0][0]
    return (ilens-jlens//16+7)%16

def make_flatlamps(
    flatlamp_dir : str,
    micropupil_dir : str,
    trace_dir : str,
    arr_mask : np.ndarray,
    filter_name:str='KN2',
    resolution:str='4000'
)->int:
    if not os.path.exists(flatlamp_dir):
        os.makedirs(flatlamp_dir)
    for k in range(16):
        shape = (1459, 128, 128)
        testcube=np.zeros(shape)
        for i in np.arange(k-7,k-7+16*9,16):
            for j in range(8):
                if 0<=i+j<128:
                    testcube[:,i+j,16*j:16*(j+1)] = np.ones_like(testcube[:,i+j,16*j:16*(j+1)])
        sim = simulate_lenslet_raw_frame(testcube.astype(np.float32),arr_mask,micropupil_dir,trace_dir,filter_name,resolution)
        fits.writeto(flatlamp_dir+filter_name+'_'+resolution+'_flatlamp'+str(k+1)+'.fits',sim['sim'])
    return 0

@njit
def extract_recmat(
    output_recmat:np.ndarray,
    output_offsets:np.ndarray,
    output_mask:np.ndarray,
    scan_array:np.ndarray,
    x1, x2, x3, x4, x5,
    y1, y2, y3, y4, y5,
    arr_mask:np.ndarray,
    width:int,
    height:int,
    pad:int
):
    nx1 = len(x1)
    for i in range(nx1):
        if (
            (x1[i] < 0)
            or (x1[i] > 4095)
            or (x5[i] < 0)
            or (x5[i] > 4095)
            or (y1[i] < 0)
            or (y1[i] > 4095)
            or (y5[i] < 0)
            or (y5[i] > 4095)
        ):
            output_mask[i]=True
            continue
        scan = lenslet_index_to_scan_index(i,arr_mask)
        xoff = max(min(x1[i]-pad,x2[i]-pad,x3[i]-pad,x4[i]-pad,x5[i]-pad,4095),0)
        yoff = max(min(y1[i]-pad,y2[i]-pad,y3[i]-pad,y4[i]-pad,y5[i]-pad,4095),0)
        output_offsets[i] = [xoff,yoff]
        spec_slice = scan_array[scan,xoff:xoff+height,yoff:yoff+width]
        #populate the recmat
        h,w = spec_slice.shape
        output_recmat[i,:h,:w] = spec_slice
    return output_recmat,output_offsets


def make_recmat(
    flatlamp_dir: str,
    trace_dir : str,
    arr_mask : np.ndarray,
    filter_name:str='KN2',
    resolution:str='4000',
    width:int=416,
    height:int=14,
    pad:int=3
)->np.ndarray:
    #Extract data
    data = np.loadtxt(trace_dir + filter_name + '_' + resolution + '.csv', delimiter=' ')
    
    x1 = data[:, 4]
    x2 = data[:, 6]
    x3 = data[:, 8]
    x4 = data[:, 10]
    x5 = data[:, 12]

    y1 = data[:, 5]
    y2 = data[:, 7]
    y3 = data[:, 9]
    y4 = data[:, 11]
    y5 = data[:, 13]
 
    midpoint = 2048
    x1 = np.round((x1) / 0.015) + midpoint
    x2 = np.round((x2) / 0.015) + midpoint
    x3 = np.round((x3) / 0.015) + midpoint
    x4 = np.round((x4) / 0.015) + midpoint
    x5 = np.round((x5) / 0.015) + midpoint

    y1 = np.round((y1) / 0.015) + midpoint
    y2 = np.round((y2) / 0.015) + midpoint
    y3 = np.round((y3) / 0.015) + midpoint
    y4 = np.round((y4) / 0.015) + midpoint
    y5 = np.round((y5) / 0.015) + midpoint

    x1 = x1.astype(int)
    x2 = x2.astype(int)
    x3 = x3.astype(int)
    x4 = x4.astype(int)
    x5 = x5.astype(int)

    y1 = y1.astype(int)
    y2 = y2.astype(int)
    y3 = y3.astype(int)
    y4 = y4.astype(int)
    y5 = y5.astype(int)

    #creating and filling arrays
    scan_array = np.zeros((16,4096,4096))
    for k in range(16):
        scan_array[k]=fits.getdata(flatlamp_dir+filter_name+'_'+resolution+'_flatlamp'+str(k+1)+'.fits')
    recmat=np.zeros((len(x1),height,width))
    offsets=np.zeros((len(x1),2))
    mask=np.zeros((len(x1)),dtype=bool)
    extract_recmat(recmat,offsets,mask,scan_array,
                   x1, x2, x3, x4, x5,
                   y1, y2, y3, y4, y5,
                   arr_mask,width,height,pad)
    
    # creating recmat file
    primary_hdu = fits.PrimaryHDU()
    hdu1 = fits.ImageHDU(data=recmat, name='RECMAT')
    hdu2 = fits.ImageHDU(data=offsets, name='OFFSETS')
    hdu3 = fits.ImageHDU(data=mask.astype(np.uint8), name='MASK')
    hdulist = fits.HDUList([primary_hdu, hdu1, hdu2, hdu3])
    hdulist.writeto(filter_name+'_'+resolution+'_'+'recmat.fits', overwrite=True)

    return recmat,offsets

@njit
def rectify(rawframe:np.ndarray,recmat:np.ndarray,offsets:np.ndarray,mask:np.ndarray,numiter:int,var:np.ndarray):
    eps = 1e-5
    dh, dw = rawframe.shape
    l,h,w = recmat.shape
    output_spectra=np.ones((l,w))*np.sum(rawframe)/np.sum(recmat)
    output_var=np.ones_like(output_spectra)*np.sum(var)/np.sum(recmat)
    tot = np.sum(recmat,axis=1)
    for ii in range(numiter):
        print("Doing iteration number",ii+1)
        #treat each column independently
        for col in range(dw):
            #prediction stage for this column
            pred = np.zeros((dh))
            for lens in range(l):
                xoff = offsets[lens,1]
                if (not mask[lens]) and (xoff<=col<xoff+w):
                    yoff = offsets[lens,0]
                    heff = min(h,dh-yoff)
                    pred[yoff:yoff+heff] += recmat[lens,:heff,col-xoff]*output_spectra[lens,col-xoff]
            #calculate ratio of rawframe to prediction
            ratio = np.zeros_like(pred)
            for row in range(dh):
                if pred[row]>eps: ratio[row] = rawframe[row,col]/pred[row]
            #iterative update of output spectra
            for lens in range(l):
                xoff = offsets[lens,1]
                if (not mask[lens]) and (xoff<=col<xoff+w):
                    if tot[lens,col-xoff]>eps:
                        yoff = offsets[lens,0]
                        heff = min(h,dh-yoff)
                        output_spectra[lens,col-xoff]*=np.sum(ratio[yoff:yoff+heff]*recmat[lens,:heff,col-xoff])/tot[lens,col-xoff]
    for col in range(dw):
        for lens in range(l):
            xoff = offsets[lens,1]
            if (not mask[lens]) and (xoff<=col<xoff+w):
                yoff = offsets[lens,0]
                heff = min(h,dh-yoff)
                output_var[lens,col-xoff] = np.sum(var[yoff:yoff+heff,dw]*recmat[lens,:heff,col-xoff])/tot[lens,col-xoff]
    output_spectra=output_spectra*tot
    output_var=output_var*tot*output_spectra
    return output_spectra, output_var

@njit
def arrange_cube(output_spectra,output_var,arr_mask,offsets,mask):
    l,w = output_spectra.shape
    output_cube = np.ones((128,128,w))*np.mean(output_spectra)
    var_cube=np.ones_like(output_cube)*np.mean(output_var)
    col_offset = np.zeros((128,128),dtype=np.int16)
    for lens in range(l):
        if not mask[lens]:
            loc=np.where(arr_mask==lens)
            ilens = loc[1][0]
            jlens = loc[0][0]
            output_cube[ilens,jlens]=output_spectra[lens]
            var_cube[ilens,jlens]=output_var[lens]
            col_offset[ilens,jlens]=offsets[lens,1]
    return output_cube,col_offset,var_cube

def extract_spectra(rawframe_fn:str,recmat_fn:str,arr_mask:np.ndarray,numiter:int=10,write:bool=True,output_fn:str='output_cube.fits'):
    rawframe=fits.getdata(rawframe_fn,extname='IM').astype(np.float64)
    var=fits.getdata(rawframe_fn,extname='VAR').astype(np.float64)
    recmat=fits.getdata(recmat_fn,extname='RECMAT').astype(np.float64)
    offsets=fits.getdata(recmat_fn,extname='OFFSETS').astype(np.int32)
    mask=fits.getdata(recmat_fn,extname='MASK').astype(np.bool_)
    output_spectra, output_var = rectify(rawframe,recmat,offsets,mask,numiter,var)
    output_cube, col_offset, var_cube = arrange_cube(output_spectra,output_var,arr_mask,offsets,mask)
    if write: 
        primary_hdu = fits.PrimaryHDU()
        hdu1 = fits.ImageHDU(data=output_cube, name='CUBE')
        hdu2 = fits.ImageHDU(data=col_offset, name='OFFSETS')
        hdu3 = fits.ImageHDU(data=mask.astype(np.uint8), name='MASK')
        hdu4 = fits.ImageHDU(data=var_cube, name='VAR')
        hdulist = fits.HDUList([primary_hdu, hdu1, hdu2,hdu3,hdu4])
        hdulist.writeto(output_fn, overwrite=True)
    return output_cube

##############################################
####### Making wavelength solutions   ########
##############################################

# 'Ground truth' from Zemax model
def make_wave_sol_from_csv(
    trace_dir : str,
    arr_mask : np.ndarray,
    filter_name:str='KN2',
    resolution:str='4000',
    degree:int =3
)->np.ndarray:
    #Extract data
    data = np.loadtxt(trace_dir + filter_name + '_' + resolution + '.csv', delimiter=' ')
    midpoint = 2048
    x1 = data[:, 4]/0.015 + midpoint
    x2 = data[:, 6]/0.015 + midpoint
    x3 = data[:, 8]/0.015 + midpoint
    x4 = data[:, 10]/0.015 + midpoint
    x5 = data[:, 12]/0.015 + midpoint

    y1 = data[:, 5]/0.015 + midpoint
    y2 = data[:, 7]/0.015 + midpoint
    y3 = data[:, 9]/0.015 + midpoint
    y4 = data[:, 11]/0.015 + midpoint
    y5 = data[:, 13]/0.015 + midpoint

    nx1 = len(x1)

    xarr = np.zeros((5))
    yarr = np.zeros((5))
    filter_data = load_filter_data(filter_name)
    wavearr = np.linspace(filter_data['min_wave'],filter_data['max_wave'],5)*1000
    midwave = wavearr[2]
    wavearr = wavearr-midwave

    wave_soln = np.zeros((135,135,degree+1))
    ywave_soln = np.zeros((135,135,degree+1))
    for i in range(nx1):
        loc = np.where(arr_mask == i)

        xarr[0] = x1[i]
        xarr[1] = x2[i]
        xarr[2] = x3[i]
        xarr[3] = x4[i]
        xarr[4] = x5[i]

        yarr[0] = y1[i]
        yarr[1] = y2[i]
        yarr[2] = y3[i]
        yarr[3] = y4[i]
        yarr[4] = y5[i]


        wave_soln[loc[0][0]+7-loc[1][0]//16,loc[1][0]+loc[0][0]//16] = np.flip(np.polyfit(wavearr,yarr,degree))
        ywave_soln[loc[0][0]+7-loc[1][0]//16,loc[1][0]+loc[0][0]//16] = np.flip(np.polyfit(wavearr,xarr,degree))

    for i in range(7):
        for j in range(7):
            ip = 22 + 16*i - j
            jp = 16 + 16*j + i
            wave_soln[ip,jp] = (wave_soln[ip-1,jp]+wave_soln[ip+1,jp]+wave_soln[ip,jp-1]+wave_soln[ip,jp+1])/4
            ywave_soln[ip,jp] = (ywave_soln[ip-1,jp]+ywave_soln[ip+1,jp]+ywave_soln[ip,jp-1]+ywave_soln[ip,jp+1])/4
    fits.writeto('data/wave_solns/'+filter_name + '_' + resolution +'_true_wave_soln.fits',wave_soln,overwrite=True)
    fits.writeto('data/wave_solns/'+filter_name + '_' + resolution +'_true_wave_soln_y.fits',ywave_soln,overwrite=True)
    return wave_soln
    

