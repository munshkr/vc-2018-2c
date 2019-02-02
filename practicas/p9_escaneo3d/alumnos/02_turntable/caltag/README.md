# CALTag


## News

2018-05-21: LAUCalTagWidget: Daniel Lau has built a real-time implementation of CALTag, available [here](https://github.com/drhalftone/LAUCalTagWidget).


## Instructions

### 1. Requirements:

  * MATLAB (tested with R2010a) with Image Processing toolbox. It has been reported broken in MATLAB 2014. You can work around the problem by copying the kmeans function from an earlier version of MATLAB, or uncommenting the "MATLAB 2014" lines in caltag.m
  * MATLAB Optimization toolbox (lsqnonlin)
  * MATLAB Statistics and Machine Learning toolbox (means)
  * Python (tested with 2.6)


### 2. Create pattern and datafile:

Run the generate_pattern.py script and convert the ps file to pdf, e.g.:

```bash
$ cd GeneratePattern
$ chmod u+x generate_pattern
$ ./generate_pattern -r 8 -c 4
```

This will produce an output.ps and output.mat file
Convert the ps to pdf for easier printing:

```bash
$ ps2pdf output.ps output.pdf
```

Print the output.pdf file, mount it to a flat surface and photograph it.


### 3. Run CALTag:

Download the photos and start Matlab. Please note that Matlab 2010a or later is
required, since parts of the code make use of the new "[~,x] = func()" syntax
to ignore some return values. These can be easily replaced with junk variables
if you wish to run on an older version of Matlab. The one other point of
incompatibility is the use of the "bwconncomp" and "regionprops" functions,
which have altered their semantics in recent versions for reduced memory use.
See
<a
href="http://blogs.mathworks.com/steve/2009/07/06/a-new-look-for-connected-component-labeling-in-r2009a/">here</a>
for details.

Add CALTag to your path

```matlab
> addpath( '/path/to/caltag' )
```

Load an image and run CALTag

```matlab
>> I = imread( 'IMG001.jpg' );
>> [wPt,iPt] = caltag( I, '/path/to/caltag/GeneratePattern/output.mat', false );
```

The second argument is the location of the datafile produced by
generate_pattern. It contains all the parameters of your particular pattern.
The third argument may be set to true if you want to see debugging information.

This will return two matrices (or nulls, if no pattern is detected). World
points are in wPt, image-space points are in iPt. These are both Nx2 matrices
where each row corresponds to a [y,x] coordinate. Note that the image points
are in Matlab-style 1-based [row,col] coordinates. There is a flag in the code
you can switch if you prefer C-style 0-based [x,y] coordinates, or else you can
just modify the return value.


## Support

This code is supported on a best-effort basis. I have no plans to make any further modifications.
