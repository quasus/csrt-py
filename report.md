# Tweaking CSRT for hand tracking

As a reference we use a Python implementation of CSRT being a modified version
of https://github.com/wwdguu/pyCFTrackers.  It is fairly close to the C++
implementation of the tracker in OpenCV.

## Tracking algorithm

The tracking involves the following steps.

1. **Feature extraction**: the patch determined by object’s previous location and
   scale is resized to a fixed size and HoG and CN features are extracted.

2. **Window response**: FFT and IFFT are used to compute the cross-correlation of
   the filter and weighted features.

3. **Localization** using the window response.

4. **Scale estimation**: the current frame is used to estimate the new scale of
   the object.  Scales from a fixed scale pool are considered separately; each
   one requires FFT and IFFT.  These Fourier transform do not use features.

5. **Binary mask** is calculated using foreground and background histograms.

6. **A posteriori feature extraction**: new location and scale are used to
   extract the features.

7. **New filter** is computed from the mask and new features. The construction of
   an optimal filter is an iterative process with each iteration involving a
   direct and inverse Fourier transform as well as matrix arithmetics.  I
   suspect that current implementation of matrix operations is suboptimal, but
   even if it is, this inefficiency may be numpy-specific and irrelevant for
   the C++ implementation.

8. **The weights are updated.**

9. **The filter is updated.**


## Profiling

As the benchmark we run the tracker on the first 100 frames of the *hand-6*
LaSOT video.


       Ordered by: internal time

       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
         3423    3.729    0.001    3.729    0.001 {built-in method numpy.fft.pocketfft_internal.execute}
          101    1.414    0.014    4.526    0.045 csrt.py:417(create_csr_filter)
          398    0.744    0.002    0.744    0.002 {method 'cumsum' of 'numpy.ndarray' objects}
          100    0.708    0.007    0.708    0.007 {imread}
         8063    0.505    0.000    0.505    0.000 {resize}
        24437    0.377    0.000    0.377    0.000 {method 'astype' of 'numpy.ndarray' objects}
          199    0.354    0.002    1.415    0.007 feature.py:219(get_features)
         6766    0.328    0.000    0.328    0.000 {getRectSubPix}
    56456/38093    0.296    0.000    5.262    0.000 {built-in method numpy.core._multiarray_umath.implement_array_function}
           99    0.289    0.003    0.289    0.003 {waitKey}
         6766    0.280    0.000    0.280    0.000 {built-in method feature._gradient.fhog}
           99    0.212    0.002    1.741    0.018 csrt.py:344(update_histograms)
          100    0.191    0.002    0.327    0.003 csrt.py:465(get_location_prior)
        17455    0.177    0.000    0.177    0.000 {method 'reduce' of 'numpy.ufunc' objects}
         7112    0.169    0.000    0.169    0.000 {filter2D}
          400    0.162    0.000    0.164    0.000 decomp_qr.py:13(safecall)
         5742    0.138    0.000    0.138    0.000 {built-in method scipy.ndimage._nd_image.min_or_max_filter1d}
          100    0.137    0.001    0.140    0.001 csrt.py:85(get_patch)
         6766    0.135    0.000    0.135    0.000 {built-in method feature._gradient.gradMag}
          199    0.117    0.001    0.881    0.004 feature.py:201(integralVecImage)



       Ordered by: cumulative time

       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
            2    0.000    0.000   13.024    6.512 main.py:1(<module>)
       1326/1    0.034    0.000   13.023   13.023 {built-in method builtins.exec}
            1    0.068    0.068   12.346   12.346 main.py:25(tracking)
           99    0.057    0.001   10.964    0.111 csrt.py:393(update)
    56456/38093    0.296    0.000    5.262    0.000 {built-in method numpy.core._multiarray_umath.implement_array_function}
          101    1.414    0.014    4.526    0.045 csrt.py:417(create_csr_filter)
         3423    0.013    0.000    3.775    0.001 pocketfft.py:51(_raw_fft)
         3423    3.729    0.001    3.729    0.001 {built-in method numpy.fft.pocketfft_internal.execute}
          198    0.030    0.000    2.082    0.011 csrt.py:273(get_features)
          807    0.005    0.000    2.035    0.003 fft_tools.py:2(fft2)
          199    0.005    0.000    2.030    0.010 csrt.py:103(get_csr_features)
         1914    0.003    0.000    1.966    0.001 <__array_function__ internals>:2(fft)
         1914    0.005    0.000    1.961    0.001 pocketfft.py:98(fft)
          705    0.005    0.000    1.901    0.003 fft_tools.py:5(ifft2)
         1509    0.002    0.000    1.832    0.001 <__array_function__ internals>:2(ifft)
         1509    0.004    0.000    1.828    0.001 pocketfft.py:192(ifft)
           99    0.212    0.002    1.741    0.018 csrt.py:344(update_histograms)
          199    0.024    0.000    1.472    0.007 feature.py:256(extract_cn_feature)
          199    0.354    0.002    1.415    0.007 feature.py:219(get_features)
           99    0.078    0.001    0.979    0.010 csrt.py:281(get_response)


The cumulative time analysis shows that the script predictably spends almost
all of the time in the method `tracking` of an auxiliary class and 88% of this
time is spent in the `update` method of the tracker, which actually performs
the tracking.  We will not analyze the remaining 12%.

The `update` method performs the tracking.  Here is the breakdown of its
subtasks:

| Subtask of `update`    | Per call |    %    |
|------------------------|----------|---------|
| `get_features`         |    0.011 |   18.99 |
| `get_response`         |    0.010 |    8.93 |
| `update_position`      |    0.000 |    0.05 |
| `update_scale`         |    0.010 |    8.75 |
| `update_histograms`    |    0.018 |   15.88 |
| `update_weights`       |    0.006 |    5.65 |
| `update_filter`        |    0.001 |    0.49 |
| `create_csr_filter`    |    0.045 |   40.46 |


## Possible ways to boost speed

The total time analysis shows that the Fourier transform is the most time
consuming function followed by `create_csr_filter` due to matrix operations.

The Fourier transforms are two-dimensional and, unless estimating the scale,
are applied to features. The computational cost may be decreased by reducing
the spatial size of the template as well as the number of features.

Possible ways to increase speed:

1.  reducing the number of scales

2.  reducing the number of iterations when the filter is constructed

3.  reducing the template size

4.  reducing the number of features

The default number of scales is 33, which seems a lot (for instance, the LADCF
paper uses 5).  Reducing it to 5 should be safe.  In the C++ implementation the
number of scales is a parameter of the tracker and can easily be changed.

The number of iterations defaults to 4 as suggested by the authors of the
original paper.  However, just 2 iterations (like in the LADCF paper) may be
sufficient and result in a smaller number of FFTs and matrix operations.
Actually, even using just the initial approximation without any iterations does
not seem catastrophic. The number of iterations can easily be changed in the
C++ implementation (the `admm_iterations` parameter).

Before applying FFT, the template is rescaled to 200x200.  This resolution may
not be necessary when a hand is tracked.  Reducing the template size to 64x64
may be adequate.

The Python implementation uses 29 channels for tracking (HOG and others)
assigning them adaptive weights.  One can apply a kind of lasso regularization
by only retaining channels with the greatest weights.  Experiments show that
dropping all but 7 (or even 5) most significant channels calculated from the
initial boundary box does not lead to a considerable loss of quality while
boosting the performance.

While a more comprehensive comparison is coming, here is a small table:

| Scales | Iterations | Size | Features |  FPS | 
| ------ | ---------- | ---- | -------- | ---- | 
|     33 |          4 |  200 |       29 |  8.5 |
|    **5** |          4 |  200 |       29 | 15.0 |
|     33 |        **2** |  200 |       29 |  9.5 |
|     33 |          4 | **64** |       29 | 13.6 |
|     33 |          4 |  200 |      **7** | 11.5 |
|    **5** |        **2** | **64** |      **7** | 37.3 |

Applying all the optimizations simultaneously does not seem to ruin the
quality, while the speed increases four times.
