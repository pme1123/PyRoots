;author: Patrick Ewing
;created: Thurs Dec 29 2016
;
;DESCRIPTION
;This is a batch script for GIMP. It runs two methods:
;1. Color enhance, which stretches saturation to cover the maximum range
;2. White balance, which remaps intensity values (RGB and grayscale only)

;This improves python pyroots object detection in mediocre quality images taken wiht a microscope, for example.

;To run:
;Place in ~/gimp-#.#/scripts
;Open and close gimp
;In shell, run:
;    cd /directory/with/images/
;    mkdir autocolor 
;    cp * autocolor && cd autocolor
;    gimp -i -b '(autocolor-whitebalance-for-microscope "*.ext")' -b '(gimp-quit 0)' 
;
;NOTE!! THIS SCRIPT WILL OVERWRITE IMAGES. COPY TO A NEW DIRECTORY BEFORE RUNNING!!
(define (autocolor-whitebalance-for-microscope pattern)  ;run color adjust and whitebalance for anything with 'pattern'
        (let* ((filelist (cadr (file-glob pattern 1))))  ;define the filelist. 1 for filename encoding. The first return is simply the number of files, so skip it.
              (while (not (null? filelist))              ;loop through filelist
                     (let* ((filename (car filelist))    ;define variables: Filename is the first item of the (current) filelist
                            (image (car (gimp-file-load RUN-NONINTERACTIVE      ;load the image
                                          filename filename)))
                            (drawable (car (gimp-image-get-active-layer          ;identify the drawable
                                            image))))
                      (plug-in-color-enhance RUN-NONINTERACTIVE image drawable)  ;run color enhance on the drawable layer of the image
                      (gimp-levels-stretch drawable)                             ;run white balance (stretch levels) on the drawable
                      (set! drawable (car (gimp-image-get-active-layer           ;make sure the drawable is the current, active layer
                                           image)))
                      (gimp-file-save RUN-NONINTERACTIVE image drawable filename filename)      ;save the drawable layer of the image as filename
                      (gimp-image-delete image))                                                ;remove the image from the workspace
                     (set! filelist (cdr filelist)))))                                          ;remove the first item from the filelist, which we just used.
