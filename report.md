# Tweaking CSRT for hand tracking

*This is a draft.*

As a reference we use a Python implementation of CSRT being a modified version
of https://github.com/wwdguu/pyCFTrackers.  It is fairly close to the C++
implementation of the tracker in OpenCV.

As the benchmark we run the tracker on the first 100 frames of the *hand-6*
LaSOT video.

       Ordered by: internal time

       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
         3423    3.695    0.001    3.695    0.001 {built-in method numpy.fft.pocketfft_internal.execute}
          101    1.432    0.014    4.521    0.045 csrt.py:392(create_csr_filter)
          100    0.659    0.007    0.659    0.007 {imread}
          398    0.654    0.002    0.654    0.002 {method 'cumsum' of 'numpy.ndarray' objects}
         8063    0.498    0.000    0.498    0.000 {resize}
           99    0.420    0.004   10.751    0.109 csrt.py:274(update)
        24437    0.409    0.000    0.409    0.000 {method 'astype' of 'numpy.ndarray' objects}
           99    0.370    0.004    0.370    0.004 {waitKey}
         6766    0.323    0.000    0.323    0.000 {getRectSubPix}
          199    0.318    0.002    1.204    0.006 feature.py:219(get_features)



       Ordered by: cumulative time

       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
            2    0.000    0.000   12.894    6.447 main.py:1(<module>)
       1326/1    0.033    0.000   12.893   12.893 {built-in method builtins.exec}
            1    0.064    0.064   12.221   12.221 main.py:25(tracking)
           99    0.420    0.004   10.751    0.109 csrt.py:274(update)
    56456/38093    0.271    0.000    5.115    0.000 {built-in method numpy.core._multiarray_umath.implement_array_function}
          101    1.432    0.014    4.521    0.045 csrt.py:392(create_csr_filter)
         3423    0.013    0.000    3.740    0.001 pocketfft.py:51(_raw_fft)
         3423    3.695    0.001    3.695    0.001 {built-in method numpy.fft.pocketfft_internal.execute}
          807    0.005    0.000    2.006    0.002 fft_tools.py:2(fft2)
         1914    0.003    0.000    1.937    0.001 <__array_function__ internals>:2(fft)


The cumulative time analysis shows that the script predictably spends almost
all of the time in the method `tracking` of an auxiliary class and 88% of this
time is spent in the `update` method of the tracker, which actually performs
the tracking.  We will not analyze the remaining 12%.

The total time analysis shows that the Fourier transform is the most time
consuming function followed by `create_csr_filter` due to matrix operations.

The Fourier transforms are two-dimensional and are applied to features. The
computational cost may be decreased by reducing the spatial size of the
template as well as the number of features.

The construction of an optimal filter is an iterative process with each
iteration involving a direct and inverse Fourier transform as well as
arithmetic operations on arrays.  I suspect that current implementation of
matrix operations is suboptimal, but even if it is, this inefficiency may be
numpy-specific and irrelevant for the C++ implementation.

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
|    *5* |          4 |  200 |       29 | 15.0 |
|     33 |        *2* |  200 |       29 |  9.5 |
|     33 |          4 | *64* |       29 | 13.6 |
|     33 |          4 |  200 |      *7* | 11.5 |
|    *5* |        *2* | *64* |      *7* | 37.3 |

Applying all the optimizations simultaneously does not seem to ruin the
quality, while the speed increases four times.

## TODO

Describe the algorithm; estimate the implementation of its parts.
