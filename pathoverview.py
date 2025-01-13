## github.com/EpiCENTR-Lab/PATHOverview
## PATHOverview was created to arrange histology images for our publication:
## www.frontiersin.org/articles/10.3389/fonc.2023.1156743/full
## If you use PATHOverview please cite it via our paper.


from math import degrees, atan2, radians, cos, sin, ceil
from ast import literal_eval
from PIL import Image, ImageDraw, ImageOps, ImageFont
import os
from pathlib import Path
import datetime
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.widgets as mwidgets
# use the nbagg backend with IPython interactive plot turned off for nice display behaviour
matplotlib.use('nbagg')
plt.ioff()

# Openslide is imported with the first slide object to allow this path to be set after module import
# With older OpenSlide install methods on windows, the path to your OpenSlide binaries must be defined 
# before first slide_obj creation.
# This is not requred/used on mac or newer OpenSlide installs.
OPENSLIDE_PATH = None

# Image manipulation functions with mpp scale retention
def fit_image(img, new_size, resize_method = Image.Resampling.LANCZOS, fill_color = None, **kwargs):
    """A wrapper for PIL thumbnail() which returns a padded image 
    of required size with retention of metadata.
    img: image to be fit
    new_size: tuple pixel size for image returned
    resize_method: PIL.Image resampling method passed to PIL thumbnail(). Default Image.Resampling.LANCZOS
    fill_color: The image will be pasted on a background of this color. Default #ffffff
    returns: downscaled PIL image with mpp scale property recalculated
    """
    w1, h1 = img.size
    if new_size[0] > w1 and new_size[1] > h1:
        msg = "Thumbnail: new_size > current size"
        raise ValueError(msg)
    img.thumbnail(new_size, resize_method)
    w2, h2 = img.size
    downscale = max(w1/w2, h1/h2)
    if fill_color == None:
        if "fill_color" in img.info:
            fill_color = img.info["fill_color"]
        # if hasattr(img, "fill_color"):
        #     fill_color = img.fill_color
        else:
            fill_color = "#ffffff"
    img2 = Image.new('RGBA', new_size, fill_color)
    #img2.fill_color = fill_color
    img2.paste(img, ((new_size[0]-w2)//2,(new_size[1]-h2)//2), None)
    img2.info = img.info
    if "mpp" in img.info:
        img2.info["mpp"] = img.info["mpp"]*downscale
    # if hasattr(img,"mpp"):
    #     img2.mpp = img.mpp*downscale
    return img2

def fit_width(img, new_size, resize_method = Image.Resampling.LANCZOS, fill_color = None, **kwargs):
    """A wrapper for PIL thumbnail() which returns an image fit to the
    required width with height cropped.padded and metadata retention.
    img: image to be fit
    new_size: tuple pixel size for image returned
    resize_method: PIL.Image resampling method passed to PIL thumbnail(). Default Image.Resampling.LANCZOS
    fill_color: The image will be pasted on a background of this color. Default #ffffff
    returns: downscaled PIL image with mpp scale property recalculated
    """
    w1, h1 = img.size
    if new_size[0] > w1:
        msg = "Thumbnail: new_size > current size"
        raise ValueError(msg)
    thumb_size = (new_size[0], h1 * w1/new_size[0])
    img.thumbnail(thumb_size, resize_method)
    w2, h2 = img.size
    downscale = w1/w2
    if fill_color == None:
        if "fill_color" in img.info:
            fill_color = img.info["fill_color"]
        # if hasattr(img, "fill_color"):
        #     fill_color = img.fill_color
        else:
            fill_color = "#ffffff"
    img2 = Image.new('RGBA', new_size, fill_color)
    #img2.fill_color = fill_color
    img2.paste(img, ((new_size[0]-w2)//2,(new_size[1]-h2)//2), None)
    img2.info = img.info
    if "mpp" in img.info:
        img2.info["mpp"] = img.info["mpp"]*downscale
    # if hasattr(img,"mpp"):
    #     img2.mpp = img.mpp*downscale
    return img2

def thumb_image(img, new_size, resize_method = Image.Resampling.LANCZOS, **kwargs):
    """A wrapper for PIL thumbnail() which returns a thumbnailed image 
    of maximum new_size with mpp scale information retention.
    img: image to be fit
    new_size: tuple pixel size for max dimensions of image returned
    resize_method: PIL.Image resampling method passed to PIL thumbnail(). Default Image.Resampling.LANCZOS
    returns: downscaled PIL image with mpp scale property recalculated
    """
    w1, h1 = img.size
    if new_size[0] > w1 and new_size[1] > h1:
        msg = "Thumbnail: new_size > current size"
        raise ValueError(msg)
    img.thumbnail(new_size, resize_method)
    w2, h2 = img.size
    downscale = max(w1/w2, h1/h2)
    if "mpp" in img.info:
        img.info["mpp"] = img.info["mpp"]*downscale
    # if hasattr(img,"mpp"):
    #     img.mpp = img.mpp*downscale
    return img

def add_scalebar(img, scale_bar = 100, sb_ratio = 50, sb_mpp = None, sb_color = "#000000", 
                 sb_pad = 3, sb_position = "bl", sb_label = False, sb_label_size = None, **kwargs):
    """Add a scalebar to image.
    Scale information is supplied or taken from image.info["mpp"].
    Scale information in retained in returned image.
    img: PIL image to add scale bar to.
    sb: length in um of scalebar. default 100um
    sb_ratio: height of scalebar relative to image height. default 50
    sb_mpp: microns per pixel scale to use. If not supplied, img.into["mpp"] will be used.
    sb_color: color of scalebar
    sb_pad: distance of scalebar from image edge (px)
    sb_position: position of scalebar (string: 'bl', 'tl', 'br', 'tr')
    sb_label: add a size label to sb (binary)
    sb_label_size: in pixels
    returns: PIL image with mpp data retained"""
    if sb_mpp is not None:
        img.info["mpp"] = mpp # should we set this??
    elif "mpp" in img.info:
        mpp = img.info["mpp"]
    # elif hasattr(img,"mpp"):
    #     mpp = img.mpp
    else:
        msg = "No scale data (mpp in PIL info or set sb_mpp)."
        raise ValueError(msg)
    draw = ImageDraw.Draw(img)
    if scale_bar in ["Auto", "auto"]:
        # aim for a round number around 1/5 width
        um_width = img.width * mpp
        sb_target = um_width // 5
        # round up to multiple of 50um
        sb_target = ceil(sb_target / 50.0) * 50.0
        # get sb length in um to 1 sig fig
        scale_bar = float('%.1g' % sb_target)
    #number of pixels in scalebar
    sb_px_x = scale_bar//mpp
    sb_px_y = max(img.height//sb_ratio,5) #ratio scale bar to image height
    if "r" in sb_position or "right" in sb_position:
        sb_x1 = img.width-sb_pad-sb_px_x
        sb_x2 = img.width-sb_pad
    else:
        sb_x1 = 0+sb_pad
        sb_x2 = sb_px_x+sb_pad
    if "t" in sb_position or "top" in sb_position:
        sb_y1 = 0+sb_pad
        sb_y2 = sb_px_y+sb_pad
    else:
        sb_y1 = img.height-sb_px_y-sb_pad
        sb_y2 = img.height-sb_pad
    #draw scalebar
    draw.rectangle(((sb_x1, sb_y1), (sb_x2, sb_y2)), fill=sb_color, outline=sb_color)
    if sb_label:
        if sb_label_size is None:
            sb_label_size = max(sb_px_y * 2, 20)
        # use the default matplotlib font
        font = ImageFont.truetype(matplotlib.font_manager.findfont(None), sb_label_size) 
        if scale_bar>=1000:
            sb_text = f"{round(scale_bar/1000,1)}mm"
        else:
            sb_text = f"{round(scale_bar)}µm"
        _, _, width, height = draw.textbbox((0,0), sb_text, font=font) 
        # returns (left, top, right, bottom) of bounding box
        if width > sb_px_x: # label is longer than the scalebar therefore will overflow image
            if "r" in sb_position or "right" in sb_position:
                anchor = "r"
                text_x = sb_x2
            else:
                anchor = "l"
                text_x = sb_x1
        else:
            anchor = "m"
            text_x = sb_x1 + (sb_x2 - sb_x1)//2
        if "t" in sb_position or "top" in sb_position:
            text_y = sb_y2
            anchor += "a"
        else:
            text_y = sb_y1
            anchor += "d"
        
        draw.text((text_x,text_y), sb_text, font=font, anchor=anchor, fill=sb_color)
        #draw.text((pad,img.height-sb_px_y-pad), sb_text, font=font, anchor="ld", fill=color)
    return img

def apply_border(img, bw = 2, b_color = "#000000", border_crop = "width", **kwargs):
    """Applies a border to supplied image by thumbnailing image 
    on a b_color background. Image mpp scale information is adjusted.
    img: image to apply border to
    bw: border width (pixels) default 2
    b_color: border color
    border_crop: method to fit image into border. default ensures width is preserved
    returns: image of same dimensions with PIL mpp data adjusted"""
    w1, h1 = img.size
    if border_crop == "width":
        # image fit by width to protect "figures are xxx um wide", non-square 
        # images will be cropped / padded in height
        img = fit_width(img, (w1-2*bw,h1-2*bw))
    elif border_crop == "fit":
        # image padded in width & height as needed
        img = fit_image(img, (w1-2*bw,h1-2*bw))
    elif border_crop == "crop":
        # crop a border width from around image
        # makes a mess of scale bar so prefer fit method
        imgb = img.crop((bw,bw,w1-bw,h1-bw))
        imgb.info = img.info 
        # if hasattr(img,"mpp"):
        #     imgb.mpp = img.mpp
        img = imgb
    else:
        img == fit_width(img, (w1-2*bw,h1-2*bw))
    img2 = Image.new('RGBA', (w1,h1), b_color)
    img2.paste(img, (bw,bw), None)
    #img.info["mpp"] is recalculated in fitting function and can be used directly here
    img2.info = img.info
    return img2
    
def apply_wb(img, wb, use_wb, **kwargs):
    wb = np.array(wb)
    if isinstance(use_wb, float):
        wb = ((wb-1) * use_wb) + 1
    new = np.clip(np.array(img)*wb[np.newaxis,np.newaxis,:],0,255)
    img2 = Image.fromarray(new.astype(np.uint8))
    img2.info = img.info
    # apply wb to fill_color
    if "fill_color" in img2.info:
        if isinstance(use_wb, float):
            img2.info["fill_color"] = tuple(np.clip(np.array(img2.info["fill_color"])*wb,0,255).astype(int))
        else:
            img2.info["fill_color"] = (255,255,255,255)
    return img2

def add_label(img, label = None, label_xpad = 5, label_ypad = 5, 
              label_position = "br", label_size = "auto", label_color = "#000000", **kwargs):
    """Add a label to image.
    img: PIL image to add label to.
    label: string to add
    label_xpad: 
    label_ypad:
    label_color: color of label
    label_position: position of label (string: 'bl', 'tl', 'br', 'tr')
    label_size: in pixels
    returns: PIL image with mpp data retained"""

    if label == None or label == "":
        return img

    if str(label_size).lower() in ["auto"]:
        label_size = img.height // 12
        # track sb position and shrink label to not overlap?
        
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(matplotlib.font_manager.findfont(None), label_size)
    
    if "r" in label_position:
        anchor = "r"
        text_x =  img.width - label_xpad
    else:
        anchor = "l"
        text_x =  label_xpad
        
    if "t" in label_position:
        anchor += "t"#"a"
        text_y =  label_ypad
    else:
        anchor += "b"#"d"
        text_y =  img.height - label_ypad

    draw.text((text_x,text_y), label, font=font, anchor=anchor, fill=label_color)
    return img
    

              
# class wrapper for openslide object with added picture output formatting functions
class slide_obj:
    openslide_imported = False
    
    def __init__(self, file, rotation = 0, mirror = False, zoom = (0.5,0.5), 
                 crop = ((0.5,0.5),1,1), mpp_x = None, wb_point = None):
        # Openslide was imported at the first object creation to allow for calling
        # os.add_dll_directory(Path(OPENSLIDE_PATH)) with older windows binaries installs
        # this is no longer required but will be used when OPENSLIDE_PATH is defined.
        if not slide_obj.openslide_imported:
            global openslide
            if OPENSLIDE_PATH is None or OPENSLIDE_PATH == "":
                import openslide
                slide_obj.openslide_imported = True
            else:
                with os.add_dll_directory(Path(OPENSLIDE_PATH)):
                    import openslide
                slide_obj.openslide_imported = True
            # if hasattr(os, 'add_dll_directory'):
            #     # Python >= 3.8 on Windows
            #     if OPENSLIDE_PATH is None or OPENSLIDE_PATH == "":
            #         msg = "Please specify path to OpenSlide\\bin"
            #         raise ModuleNotFoundError(msg)
            #     with os.add_dll_directory(Path(OPENSLIDE_PATH)):
            #         import openslide
            # else:
            #     import openslide
            # slide_obj.openslide_imported = True
        self.filename = Path(file)
        #self.slide = openslide.OpenSlide(self.filename)
        # This should include support for image files with similar api
        # https://openslide.org/api/python/#wrapping-a-pil-image
        self.slide = openslide.open_slide(self.filename)
        if "openslide.mpp-x" in self.slide.properties.keys():
            self.mpp_x = float(self.slide.properties["openslide.mpp-x"])
        elif mpp_x != None:
            self.mpp_x = mpp_x
        else:
            msg = "mpp_x value not set in slide file, specify mpp_x with object creation"
            raise ValueError(msg)
        self.rotation = float(rotation) if pd.notnull(rotation) else 0
        self.mirror = mirror if pd.notnull(mirror) else False
        self.crop = eval(str(crop)) if pd.notnull(crop) else ((0.5,0.5),1,1)
        self.zoom_point = eval(str(zoom)) if pd.notnull(zoom) else (0.5,0.5)
        self.wb_point = eval(str(wb_point)) if pd.notnull(wb_point) else None
        self.fill_color = self._get_fill_color(self.wb_point)
        self.gamma = 1.0 # Experimental
        self.wb = self._get_wb()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exception_type, exception_value, exception_traceback) :
        self.close()
        
    def close(self):
        self.slide.close()
        self.slide = None
    
    def _get_fill_color(self, wb_point = "side", wb_sample_width = 10):
        """ Calculate an average color (median of each chanel) from the edge 
            of the slide which is assumed to be a blank background. This is 
            used for filling the edges of the image on crop and rotation and 
            white balance calculation.
            Default is the median px of a 10px strip from both the left and right of the slide.
            A point (wb_point) can be specified from which a 300um x 300um square is sampled.
            """
        if wb_point in ["Top", "top"]:
            # This uses a single pixel strip from the top of the slide
            # Not great results with OS-1 as image extends beyond glass
            # so use left and right for higher chance of clean background
            
            # Calculate downsample for >=1000 px strip
            downsample = self.slide.dimensions[0]/1000
            level = self.slide.get_best_level_for_downsample(downsample)
            temp_img = self.slide.read_region((0, 0), level, (self.slide.level_dimensions[level][0],1))

            # Find an average pixel across this strip. Here using 50 percentile so can adjust.
            fill_color = tuple(np.percentile(np.array(temp_img)[0,:], 50, axis=0).astype(int))
        elif wb_point is not None and (type(wb_point) == tuple or "(" in wb_point):
            wb_point = eval(str(wb_point)) # from excel will be str
            ### get a 30x30 image of 300um at wb_point
            img = self._get_sub_image((30,30), image_centre = wb_point, 
                                      true_size = 300, use_wb = False, 
                                      gamma = 1, fill_color = "#ffffff")
            img_data = np.array(img)
            fill_color = tuple(np.percentile(img_data[:,:], 50, axis=(0,1)).astype(int))
        else: # default sides of slide
            # Calculate downsample for >=1000 px strip
            downsample = self.slide.dimensions[1]/1000
            level = self.slide.get_best_level_for_downsample(downsample)
            level_downsample = self.slide.level_downsamples[level]
            
            # Get left hand strip of width wb_sample_width
            img_data = np.array(self.slide.read_region((0, 0), 
                                                           level, (wb_sample_width,self.slide.level_dimensions[level][1])))
            # Get right hand strip of width wb_sample_width
            img_data2 = np.array(self.slide.read_region((self.slide.dimensions[0]-int(wb_sample_width*level_downsample), 0), 
                                                            level, (wb_sample_width,self.slide.level_dimensions[level][1])))
            # Append two samples for averaging
            img_data = np.append(img_data, img_data2, axis = 1)

            # Find median pixel intensity (percentile 50 so can tinker with %)
            fill_color = tuple(np.percentile(img_data[:,:], 50, axis=(0,1)).astype(int))
            
        return fill_color

    def _get_wb(self):
        """ Calculate a float for each chanel to multiply fill_color up to white.
            Returns a tuple of floats for RGBA
        """
        # Find the factor to multiply each chanel up to 255
        wb = tuple(255/i for i in self.fill_color)
        return wb
    
    def get_raw_overview_image(self, image_size = (500,500), scale_bar=None, **kwargs):
        """ Return a PIL.Image containing an RGBA thumbnail of the slide.
            Rotations and crop are not applied.
            size:       overall dimensions of the image.
            scale_bar:  size of scalebar in um.
        """
        downsample = max(*(dim / thumb for dim, thumb in zip(self.slide.dimensions, image_size)))
        level = self.slide.get_best_level_for_downsample(downsample)
        tile = self.slide.read_region((0, 0), level, self.slide.level_dimensions[level])

        img = Image.new('RGBA', tile.size, "#ffffff")
        img.paste(tile, None, None)
        draw = ImageDraw.Draw(img)

        #calculate um per pixel at this downsample
        mpp_x = self.mpp_x * self.slide.level_downsamples[level]
        #img.mpp = mpp_x
        img.info["mpp"] = mpp_x
        img = thumb_image(img, image_size)
        if scale_bar:
            img = add_scalebar(img, scale_bar=scale_bar, **kwargs)        
        return img

    def get_macro_image(self, max_size = 500):
        """ Return PIL.Image containing the macro slide image thumbnailed to max_size
            if available else a 2x2 blank image.
        """
        if "macro" in self.slide.associated_images.keys():
            return thumb_image(self.slide.associated_images["macro"], (max_size,max_size))
        else:
            return Image.new('RGBA', (2,2), "#ffffff")

    def get_label_image(self, max_size = 500, rotated = True):
        """ Returns a PIL.Image of macro slide image cropped to a square containing the label
        """
        img = self.get_macro_image(max_size)
        if rotated:
            return img.crop((0,0,img.height,img.height)).rotate(-90)
        else:
            return img.crop((0,0,img.height,img.height))
    
    def _get_sub_image(self, image_size, image_centre = (0.5,0.5), 
                      true_size = None, rotation = None, mirror = None, scale_bar = None, 
                        fill_color = "auto", gamma = None, wb = None, use_wb = False, **kwargs):
        image_width = image_size[0] #300 #px
        image_height = image_size[1]
        if true_size is None:
            true_size = tuple(i * self.mpp_x for i in self.slide.dimensions)
        elif type(true_size) == int or type(true_size) == float:
            # if one dimension given calculate second at correct ratio
            true_size = (true_size,true_size*(image_height/image_width))
        true_width = true_size[0]
        true_height = true_size[1]

        target_mpp = true_width/image_width
        target_downsample = target_mpp / self.mpp_x
        level = self.slide.get_best_level_for_downsample(target_downsample)
        level_downsample = self.slide.level_downsamples[level]

        level_mpp = level_downsample * self.mpp_x
        
        # get tile of double largest side centred on zoom_point, later crop out centre
        tile_width = int(max(true_width, true_height)*2/level_mpp)
        tile_x = int((self.slide.dimensions[0]*image_centre[0]) - tile_width*level_downsample/2)
        tile_y = int((self.slide.dimensions[1]*image_centre[1]) - tile_width*level_downsample/2)
        
        tile = self.slide.read_region((tile_x,tile_y), level, (tile_width,tile_width))
        
        # tile comes with zero alpha in areas beyond the image dimension 
        # (we've oversampled to double size to allow cropping) 
        # so if fill_color = auto, paste onto image with slides neutral 
        # background color calculated on creation
        if fill_color is None:
            fill_color = "#ffffff"
        elif fill_color in ["auto","Auto"]:
            fill_color = self.fill_color
        
        img = Image.new('RGBA', tile.size, fill_color)
        img.info["fill_color"] = fill_color
        img.paste(tile, (0,0), tile) # (image, position, mask: using the tile alpha)
        
        if mirror:
            img = ImageOps.mirror(img)
        
        if rotation:
            img = img.rotate(rotation, fillcolor=fill_color, expand=False)
        
        #crop image to required dimensions
        crop_width = true_width/level_mpp
        crop_height = true_height/level_mpp
        w, h = img.size
        crop_x = (w-crop_width)/2
        crop_y = (h-crop_height)/2
        img = img.crop((crop_x, crop_y, w-crop_x, h-crop_y))
        
        #img.mpp = level_mpp
        img.info["mpp"] = level_mpp
        img = fit_image(img, image_size, fill_color=fill_color)
        
        # Experimental
        if use_wb is not False:
            if wb is None:
                wb = self.wb
            img = apply_wb(img, wb, use_wb)
        
        # Experimental
        # apply gamma on final, background filled image before scalebar
        if gamma is None:
            gamma = self.gamma
        if gamma != 1:
            # apply gamma correction.
            # This creates LUT for all possible values then applies
            img = img.point(lambda p: pow(p/255, (1/gamma))*255)
        
        if scale_bar:
            img = add_scalebar(img, scale_bar=scale_bar, **kwargs)
        return img
    
    def get_zoom_image(self, image_size, true_width, zoom_point = None, 
                       rotation = None, mirror = None, **kwargs):
        """ Returns a PIL.Image 
        """
        if zoom_point is None:
            zoom_point = self.zoom_point
        if rotation is None:
            rotation = self.rotation
        if mirror is None:
            mirror = self.mirror
        return self._get_sub_image(image_size, image_centre = zoom_point, 
                                  true_size = true_width, rotation = rotation, mirror = mirror, 
                                   **kwargs)
    
    def get_crop_image(self, image_size, rotation = None, mirror = None, crop = None, crop_real_width = None, **kwargs):
        """ Returns PIL.Image of cropped and rotated ROI from slide
            required:
            image_size
            if other parameters are None, the slide_obj properties are used.
        """
        if rotation is None:
            rotation = self.rotation
        if mirror is None:
            mirror = self.mirror
        if crop is None:
            crop = self.crop
        if crop_real_width is not None:
            return self._get_sub_image(image_size, image_centre = crop[0], 
                                  true_size = crop_real_width,
                                  rotation = rotation, mirror = mirror, **kwargs)
        return self._get_sub_image(image_size, image_centre = crop[0], 
                                  true_size = self._relative_to_true((crop[1],crop[2])),
                                  rotation = rotation, mirror = mirror, **kwargs)
    
    def get_overview_image(self, image_size, rotation = None, mirror = None, **kwargs):
        """ Returns PIL.Image of whole slide with rotation and mirroring
            required:
            image_size
            if other parameters are None, the slide_obj properties are used.
        """
        if rotation is None:
            rotation = self.rotation
        if mirror is None:
            mirror = self.mirror
        return self._get_sub_image(image_size,
                                  rotation = rotation, mirror = mirror, **kwargs)
    
    # def _relative_to_l0(self, rel):
    #     return (rel[0]*self.slide.dimensions[0], rel[1]*self.slide.dimensions[1])
    
    def _relative_to_true(self, rel):
        return (rel[0]*self.slide.dimensions[0]*self.mpp_x, rel[1]*self.slide.dimensions[1]*self.mpp_x)

    def ndpa_to_relative(self, ndpa_point):
        """
        NDPA file co-ordinates are stored in nm from physical slide centre.
        Slide centre to overview image centre is stored in:
            hamamatsu.XOffsetFromSlideCentre
            hamamatsu.YOffsetFromSlideCentre
        This takes an NDPA point (ie nm from physical slide centre) and
        returns the point relative to overview image (ie fraction of image 
        from top left).
        """
        # nm from top left
        # top left in nm
        wnm = self.slide.dimensions[0] * self.mpp_x * 1000
        hnm = self.slide.dimensions[1] * self.mpp_x * 1000
        leftnm = int(self.slide.properties["hamamatsu.XOffsetFromSlideCentre"]) - (wnm / 2)
        topnm = int(self.slide.properties["hamamatsu.YOffsetFromSlideCentre"]) - (hnm / 2)
        return ((ndpa_point[0] - leftnm) / wnm, (ndpa_point[1] - topnm) / hnm)

    def relative_to_ndpa(self, relative_point):
        """
        NDPA file co-ordinates are stored in nm from physical slide centre.
        Slide centre to overview image centre is stored in:
            hamamatsu.XOffsetFromSlideCentre
            hamamatsu.YOffsetFromSlideCentre
        This takes a point relative to overview image (ie fraction of image 
        from top left) and returns an NDPA point (ie nm from physical slide 
        centre).
        """
        # nm from top left
        # top left in nm
        wnm = self.slide.dimensions[0] * self.mpp_x * 1000
        hnm = self.slide.dimensions[1] * self.mpp_x * 1000
        leftnm = int(self.slide.properties["hamamatsu.XOffsetFromSlideCentre"]) - (wnm / 2)
        topnm = int(self.slide.properties["hamamatsu.YOffsetFromSlideCentre"]) - (hnm / 2)
        return ((relative_point[0] * wnm) + leftnm, (relative_point[1] * hnm) + topnm)
        

    def _rotate_point(self, loc1, rototation = 0, centre = (0,0)):
        rads = radians(rototation)
        new_x = centre[0] + cos(rads) * (loc1[0] - centre[0]) - sin(rads) * (loc1[1] - centre[1])
        new_y = centre[1] + sin(rads) * (loc1[0] - centre[0]) + cos(rads) * (loc1[1] - centre[1])
        #loc2 = (new_x/image1.size[0], new_y/image1.size[1])
        return (new_x, new_y)
    
    def get_figure(self, panel_size = (500,500), add_inset = True, inset_size = None, zoom_real_size = 250, 
                   rotation = None, mirror = None, zoom_point = None, crop = None, 
                   scale_bar = None, inset_scale_bar = None, crop_real_width = None, 
                   **kwargs):
        """
        Returns PIL.Image containing cropped overview image with inset zoom image (add_inset = True).
        If parameters are None, slide_obj parameters are used.
        """
        if rotation is None:
            rotation = self.rotation
        if mirror is None:
            mirror = self.mirror
        if crop is None:
            crop = self.crop
        if zoom_point is None:
            zoom_point = self.zoom_point
        if inset_size is None:
            # calculate inset to be square of min 1/2.5 width or 1/2.5 height of the panel
            inset_dim = int(min(panel_size[0], panel_size[1])/2.5)
            inset_size = (inset_dim, inset_dim)

        base_image = self.get_crop_image(
                                        panel_size, rotation = rotation, mirror = mirror, 
                                        crop = crop, scale_bar = scale_bar, crop_real_width = crop_real_width,
                                        **kwargs)

        base_image = add_label(base_image, **kwargs)
        base_image = apply_border(base_image, **kwargs)
        
        if add_inset:
            zoomed_image = apply_border(self.get_zoom_image(
                                                inset_size, zoom_real_size, zoom_point = zoom_point, 
                                                rotation = rotation, mirror = mirror, 
                                                scale_bar = inset_scale_bar, **kwargs), **kwargs)
            
            relative_zoom_point = tuple(i-j for i, j in zip(zoom_point,crop[0]))
            if mirror:
                relative_zoom_point = (-relative_zoom_point[0],relative_zoom_point[1])
            # calculate um distance between middle of crop and zoom point on non-rotated 
            # image as is relative to non-rotated scan dimensions
            zoom_point_offset = (relative_zoom_point[0]*self.slide.dimensions[0]*self.mpp_x,
                                 relative_zoom_point[1]*self.slide.dimensions[1]*self.mpp_x)
            zoom_point_offset = self._rotate_point(zoom_point_offset, -rotation, (0,0))

            box_x = zoom_point_offset[0] / base_image.info["mpp"] + 0.5*base_image.width
            box_y = zoom_point_offset[1] / base_image.info["mpp"] + 0.5*base_image.height
            box_width = zoom_real_size/base_image.info["mpp"]
            box_height = box_width/inset_size[0]*inset_size[1]

            draw = ImageDraw.Draw(base_image)
            draw.rectangle((
                (int((box_x-box_width/2)),
                 int((box_y-box_height/2))),
                (int((box_x+box_width/2)),
                 int((box_y+box_height/2)))
                ), outline="black")

            base_image.paste(zoomed_image, (int(base_image.width-zoomed_image.width),0))
            del zoomed_image

        return base_image
    
    def get_figure_inverted(self, panel_size = (500,500), add_inset = True, 
                    inset_size = None, zoom_real_size = 250, 
                    rotation = None, mirror = None, zoom_point = None, crop = None, 
                    scale_bar = None, inset_scale_bar = None, crop_real_width = None, **kwargs):
        """
            Returns PIL.Image containing cropped overview image with inset zoom image (add_inset = True).
            If parameters are None, slide_obj parameters are used.
        """
        if rotation is None:
            rotation = self.rotation
        if mirror is None:
            mirror = self.mirror
        if crop is None:
            crop = self.crop
        if zoom_point is None:
            zoom_point = self.zoom_point
        if inset_size is None:
            # calculate inset to be square of min 1/2.5 width or 1/2.5 height of the panel
            inset_dim = int(min(panel_size[0], panel_size[1])/2.5)
            inset_size = (inset_dim, inset_dim)
        
        zoomed_image = self.get_zoom_image(panel_size, zoom_real_size, zoom_point = zoom_point, 
                                            rotation = rotation, mirror = mirror, scale_bar = scale_bar, 
                                            **kwargs)

        zoomed_image = add_label(zoomed_image, **kwargs)
        zoomed_image = apply_border(zoomed_image, **kwargs)
        
        if add_inset:
            base_image = apply_border(self.get_crop_image(
                                                inset_size, rotation = rotation, mirror = mirror, 
                                                crop = crop, scale_bar = inset_scale_bar, 
                                                crop_real_width = crop_real_width,
                                                **kwargs), **kwargs)
        
            relative_zoom_point = tuple(i-j for i, j in zip(zoom_point,crop[0]))
            if mirror:
                relative_zoom_point = (-relative_zoom_point[0],relative_zoom_point[1])
            # calculate um distance between middle of crop and zoom point on non-rotated image 
            # as is relative to non-rotated scan dimensions
            zoom_point_offset = (relative_zoom_point[0]*self.slide.dimensions[0]*self.mpp_x,
                                 relative_zoom_point[1]*self.slide.dimensions[1]*self.mpp_x)
            zoom_point_offset = self._rotate_point(zoom_point_offset, -rotation, (0,0))

            box_x = zoom_point_offset[0] / base_image.info["mpp"] + 0.5*base_image.width
            box_y = zoom_point_offset[1] / base_image.info["mpp"] + 0.5*base_image.height
            box_width = zoom_real_size/base_image.info["mpp"]
            box_height = box_width/panel_size[0]*panel_size[1]

            draw = ImageDraw.Draw(base_image)
            draw.rectangle((
                (int((box_x-box_width/2)),
                 int((box_y-box_height/2))),
                (int((box_x+box_width/2)),
                 int((box_y+box_height/2)))
                ), outline="black")
            zoomed_image.paste(base_image, (int(zoomed_image.width-base_image.width),0))

        return zoomed_image
    
    def get_summary_figure(self, width = 500):
        """ Returns an image containing the slide macro image and raw_overview. """
        raw_overview = self.get_raw_overview_image(image_size = (width,width))
        macro = self.get_macro_image(max_size = 500)
        img = Image.new('RGBA', (max(raw_overview.width,macro.width), raw_overview.height+macro.height), "#ffffff")
        img.paste(macro, (0,0), None)
        img.paste(raw_overview, (0,macro.height), None)
        return img
        
    def get_focus_check(self, size = 900, num_img = (9,9), level = 0):
        """ 
        An experimental function to return an image of num_image tiles
        evenly spaced across the specified level. Could be used to spot
        out of focus regions.
        """
        sub_size = size//num_img[0]
        img = Image.new('RGBA', (sub_size*num_img[0],sub_size*num_img[1]), "#ffffff")
        for r in range(num_img[0]):
            for c in range(num_img[1]):
                tile_x = self.slide.dimensions[0]//(num_img[0]+1)*(r+1)-sub_size//2
                tile_y = self.slide.dimensions[1]//(num_img[1]+1)*(c+1)-sub_size//2
                tile = self.slide.read_region((tile_x,tile_y), level, (sub_size,sub_size))
                img.paste(tile, (r*sub_size,c*sub_size), None)
        return img
        
# end slide_obj class


class pathofigure:
    pages = []
    
    fig_defaults = {
        "fig_type": None,
        "panel_size": (500,500),
        "scale_bar": "auto",
        "sb_label": True,
        "sb_label_size": None,
        "crop_real_width": None,
        "fill_color": "auto",
        "use_wb": True,

        "add_inset": True,
        "inset_size": None,
        "zoom_real_size": 250,
        "inset_scale_bar": "auto",

        "label": None,
        "label_size": "auto",
        "label_position": "br", #bottom right
        "label_ypad": 5, #px
        "label_xpad": 5,
        "label_color": "#000000",

        "figsize": (8.27,11.69), #A4 in inches
        "n_x": 4,
        "n_y": 6,
        "fig_layout": "compressed", 
        "dpi": 300,

        # str.format(**globals()) is applied at time of use.
        "title1": "",
        "footer": "Created with PATHOverview. github.com/EpiCENTR-Lab/PATHOverview",
        
        # Ensure missing values are None not np.nan
        "rotation": None,
        "mirror": None,
        "crop": None, 
        "zoom_point": None,
        "wb_point": None,
        "root": None,
        "file": None,
        "title2": None,
        "title3": None,
        "mpp_x": None, # must specify mpp_x scale for creating object from static tif.
    }

    @staticmethod
    def fig_from_df(df, page_row):#title = None, footer = None):
        defaults = pd.Series(pathofigure.fig_defaults)#.dropna()        
        page_row = page_row.reindex(page_row.index.union(defaults.index))
        page_row = page_row.fillna(defaults.dropna())
        page_row = page_row.replace(np.nan, None)
        
        n_x = int(page_row["n_x"])
        n_y = int(page_row["n_y"])
        figsize = literal_eval(str(page_row["figsize"]))
        
        if len(df) > (n_y * n_x):
            msg = "Too many panels for layout!"
            raise ValueError(msg)

        fig, axs = plt.subplots(n_y, n_x, layout=page_row["fig_layout"],
                                figsize=figsize, dpi=page_row["dpi"], squeeze=False)
        for ax in axs.ravel():
            ax.axis('off')

        title_list = [str(t) for t in 
                          [page_row.get("title1"),page_row.get("title2"),page_row.get("title3")] 
                          if pd.notnull(t)]
        title = "\n".join(title_list).format(**globals())
        fig.suptitle(title)
        
        if page_row["footer"]:
            footer = page_row["footer"].format(**globals())
            fig.supxlabel(footer)

        for index, row in df.iterrows():
            ax = axs.ravel()[row.get("order")]
            img = pathofigure.panel_from_row(row)
            ax.imshow(img)
            img.close()
            del img

                # # Use literal_eval to transform any parameters that have been imported from Excel
                # # as str. Force to str to handle any non-Excel data.
                # to_eval = ["rotation", "crop", "panel_size", "inset_size", "zoom_point", "wb_point"]
                # for k in to_eval:
                #     if pd.notnull(row[k]):
                #         row[k] = literal_eval(str(row[k]))
                # with slide_obj(Path(row.get("root",""),row.get("file")), wb_point = row["wb_point"]) as sld:
                    
                #     if row["fig_type"] in ["inverted","Inverted"]:
                #         # Send all of row (unpacked) to be passed on to sub-functions.
                #         # This means parameters like border width get passed on.
                #         image = sld.get_figure_inverted(**row)
                    
                #     elif row["fig_type"] in ["raw","Raw"]:
                #         image = apply_border(sld.get_raw_overview_image(
                #             image_size = row["panel_size"], sb=row["scale_bar"]))
                    
                #     elif row["fig_type"] in ["slide","Slide"]:
                #         image = apply_border(sld.get_summary_figure(
                #             width = row["panel_size"][0]))
                    
                #     else:
                #         image = sld.get_figure(**row)
                        
            #         ax.imshow(image)
            #         image.close()
            #         del image
                    
            # if pd.notnull(row.get("label")):
            #     ax.set_title(row["label"], y=row["label_y"], loc=row["label_location"],
            #                  pad=row["label_ypad"], wrap=True, fontsize = row["label_size"])
                
        return fig

    @staticmethod
    def panel_from_row(row):
        defaults = pd.Series(pathofigure.fig_defaults)
        # this doesn't work on a df (can't fill with tuple) so running here on series
        row = row.reindex(row.index.union(defaults.index))
        row = row.fillna(defaults.dropna())
        row = row.replace(np.nan, None)
        
        # Use literal_eval to transform any parameters that have been imported from Excel
        # as str. Force to str to handle any non-Excel data.
        to_eval = ["rotation", "crop", "panel_size", "inset_size", "zoom_point", "wb_point"]
        for k in to_eval:
            if pd.notnull(row[k]):
                row[k] = literal_eval(str(row[k]))
        
        if pd.notnull(row.get("file")):
            with slide_obj(Path(row.get("root",""),row.get("file")), wb_point = row["wb_point"], mpp_x = row["mpp_x"]) as sld:
                
                if row["fig_type"] in ["inverted","Inverted"]:
                    # Send all of row (unpacked) to be passed on to sub-functions.
                    # This means parameters like border width get passed on.
                    image = sld.get_figure_inverted(**row)
                
                elif row["fig_type"] in ["raw","Raw"]:
                    image = apply_border(sld.get_raw_overview_image(
                        image_size = row["panel_size"], sb=row["scale_bar"]))
                
                elif row["fig_type"] in ["slide","Slide"]:
                    image = apply_border(sld.get_summary_figure(
                        width = row["panel_size"][0]))
                
                else:
                    image = sld.get_figure(**row)
        return image

    
    # @staticmethod
    # def overview_page(df, title = None, pgnum = None):
    #     fig, axs = plt.subplots(4, 4, 
    #                             constrained_layout=True, figsize=(8.27,11.69), dpi=300)
    #     for ax in axs.ravel():
    #         ax.axis('off')
    #     if title: fig.suptitle(title)
    #     if pgnum: fig.supxlabel(pgnum)
    #     n=0
    #     for index, row in df.iterrows():
    #         ax = axs.ravel()[n]
    #         ax.set_title(row["label"])
    #         with slide_obj(Path(row['root'],row['file'])) as sld:
    #             image = sld.get_summary_figure()
    #             ax.imshow(image)
    #             image.close()
    #         n+=1
    #         if n == 16:
    #             break
    #     return fig
        
    pdf_title = "Histopathology summary images"
    pdf_author = "Created with PATHOverview: github.com/EpiCENTR-Lab/PATHOverview"
    pdf_subject = None
    pdf_keywords = "Created with PATHOverview: github.com/EpiCENTR-Lab/PATHOverview"
    pdf_creationdate = str(datetime.date.today())

    @staticmethod
    def save_pdf(figs, filename, dpi = "figure"):
        with PdfPages(Path(filename)) as pdf:
            for f in figs:
                pdf.savefig(f, dpi = dpi)
            d = pdf.infodict()
            d['Title'] = pathofigure.pdf_title
            d['Author'] = pathofigure.pdf_author
            d['Subject'] = pathofigure.pdf_subject
            d['Keywords'] = pathofigure.pdf_keywords
            d['CreationDate'] = pathofigure.pdf_creationdate
    
    @staticmethod
    def make_pages(pages_df, slides_df, pages = None, save_pages = True, retain_pages = True, save_pdf = True):
        pass
# end pathofigure

class pathoverview_interactive_fig:
    def __init__(self):#, filename, rotation = 0, mirror = False, zoom = (0,0), crop = None):
        with plt.ioff():
            self.fig = plt.figure(figsize=(8,8))
        self.fig.canvas.toolbar_visible = False
        self.ax = self.fig.add_subplot()#1, 1, 1)
        # don't load image here, use placeholder image then call load image after load data
        self.image = Image.new('RGBA', (500,500), "#000000")
        self.rotation = 0
        self.expand_rotation = True
        self.mirror = False
        # middle of the zoom image relative to the center of image1
        self.zoom_point = (0,0)
        self.crop = None
        self.crop_bounds = None
        self.width = self.image.width
        self.height = self.image.height
        self.zoom_dot = None
        self.centre = (0,0)
        #self.load_image(filename, rotation, mirror, zoom, crop)
        self.draw_fig()
        self.click_listen = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self):
        self.fig.close()

    # def load_image(self, filename, rotation = 0, mirror = False, zoom = (0,0), crop = None, mpp_x = None):
    #     with slide_obj(Path(filename), mpp_x = mpp_x).get_raw_overview_image() as img:
    #         self.image = img
        # self.rotation = rotation
        # self.expand_rotation = True
        # self.mirror = mirror
        # # middle of the zoom image relative to the center of image1
        # self.zoom_point = zoom
        # self.crop = crop
        # self.crop_bounds = None
        # self.width = self.image.width
        # self.height = self.image.height
        # self.zoom_dot = None
        # self.centre = (0,0)

    def load_slide(self, row, update = True):
        self.reset_fig()
        if Path(row['root'], row['file']) is not None:
            with slide_obj(Path(row['root'], row['file']), mpp_x = row.get("mpp_x", None)).get_raw_overview_image() as img:
                self.image = img
                self.height = self.image.height
                self.width = self.image.width
        # un pandas these:
        if pd.notnull(row["rotation"]): self.rotation = float(row["rotation"])
        if pd.notnull(row["mirror"]): self.mirror = row["mirror"]
        # excel import will be str, direct from interactive will be tuple. Force to str then eval to tuple
        if pd.notnull(row["crop"]): 
            crop = eval(str(row["crop"]))
            crop_point = self.relative_to_point(crop[0])
            self.crop = (crop_point, crop[1], crop[2])
            crop_plot = self.rotate_from_image(crop_point)
            width = crop[1]*self.width
            height = crop[2]*self.height
            # (xmin, xmax, ymin, ymax) 
            self.crop_bounds = (
                crop_plot[0]-(width/2),
                crop_plot[0]+(width/2),
                crop_plot[1]-(height/2),
                crop_plot[1]+(height/2))
        # make this correct for plotting
        if pd.notnull(row["zoom_point"]): 
            zoom_point = eval(str(row["zoom_point"]))            
            zoom_point = self.relative_to_point(zoom_point)
            self.zoom_point = zoom_point
        # self.update_fig()
    
    # def load_data(self,row):
    #     if pd.notnull(row["rotation"]): self.rotation = float(row["rotation"])
    #     if pd.notnull(row["mirror"]): self.mirror = row["mirror"]
    #     # excel import will be str, direct from interactive will be tuple. Force to str then eval to tuple
    #     if pd.notnull(row["crop"]): 
    #         crop = eval(str(row["crop"]))
    #         crop_point = self.relative_to_point(crop[0])
    #         self.crop = (crop_point, crop[1], crop[2])
    #         crop_plot = self.rotate_from_image(crop_point)
    #         width = crop[1]*self.width
    #         height = crop[2]*self.height
    #         # (xmin, xmax, ymin, ymax) 
    #         self.crop_bounds = (
    #             crop_plot[0]-(width/2),
    #             crop_plot[0]+(width/2),
    #             crop_plot[1]-(height/2),
    #             crop_plot[1]+(height/2))
    #     # make this correct for plotting
    #     if pd.notnull(row["zoom_point"]): 
    #         zoom_point = eval(str(row["zoom_point"]))            
    #         zoom_point = self.relative_to_point(zoom_point)
    #         self.zoom_point = zoom_point
    #     # self.update_fig()

    # @output.capture()
    def draw_fig(self):
        image2 = self.image
        if self.mirror:
            image2 = ImageOps.mirror(image2)
        image2 = image2.rotate(self.rotation, fillcolor="#ffffff", expand=self.expand_rotation)
        # show the image with figure origin at centre
        pic = self.ax.imshow(image2, cmap='gray',
                            extent=[-image2.width/2., image2.width/2., image2.height/2., -image2.height/2. ])
        self.ax.xaxis.set_ticks_position('top')
        # add some whitespace around the image for cropping
        fig_lim = max(self.width, self.height)//4*3
        self.ax.set_xlim(-fig_lim, fig_lim)
        self.ax.set_ylim(fig_lim, -fig_lim) #origin set to top left for images so y = backwards
        self.centre_dot = self.ax.scatter(self.centre[0],self.centre[1], color="r")
        zoom_loc = self.rotate_from_image(self.zoom_point)
        self.zoom_dot = self.ax.scatter(zoom_loc[0], zoom_loc[1], marker="x", color="k")
        self.r_selector = mwidgets.RectangleSelector(
            self.ax, self.rect_callback, interactive=True, ignore_event_outside=True, 
            use_data_coordinates=True, button = 1)
        if self.crop_bounds is not None:
            self.r_selector.set_visible(True)
            self.r_selector.extents = self.crop_bounds
        self.ax.figure.canvas.draw()
        self.fig.canvas.flush_events()
    
    def update_fig(self):
        self.ax.cla()
        self.draw_fig()
        
    def rotate_from_image(self, loc1, centre = (0,0)):
        rads = -radians(self.rotation)
        new_x = centre[0] + cos(rads) * (loc1[0] - centre[0]) - sin(rads) * (loc1[1] - centre[1])
        new_y = centre[1] + sin(rads) * (loc1[0] - centre[0]) + cos(rads) * (loc1[1] - centre[1])
        return (new_x, new_y)
    
    def rotate_to_image(self, loc1, centre = (0,0)):
        rads = radians(self.rotation)
        new_x = centre[0] + cos(rads) * (loc1[0] - centre[0]) - sin(rads) * (loc1[1] - centre[1])
        new_y = centre[1] + sin(rads) * (loc1[0] - centre[0]) + cos(rads) * (loc1[1] - centre[1])
        return (new_x, new_y)

    def point_to_relative(self, point):
        if self.mirror:
            return ((-point[0]+self.width/2)/self.width, (point[1]+self.height/2)/self.height)
        else:
            return ((point[0]+self.width/2)/self.width, (point[1]+self.height/2)/self.height)
    
    def relative_to_point(self, rel):
        if self.mirror:
            return ((-rel[0]*self.width-self.width/2), (rel[1]*self.height-self.height/2))
        else:
            return ((rel[0]*self.width-self.width/2), (rel[1]*self.height-self.height/2))

    def disconnect(self):
        """ Uninstall the event handlers for the plot. """
        for connection in self.connections:
            self.fig.canvas.mpl_disconnect(connection)

    def rect_callback(self, eclick, erelease):
        # store the current location for re-drawing
        self.crop_bounds = self.r_selector.extents
    
    def get_rect(self):
        if self.crop_bounds is not None:
            width = (self.r_selector.extents[1] - self.r_selector.extents[0])/self.width
            height = (self.r_selector.extents[3] - self.r_selector.extents[2])/self.height
            #((center), x width, y height)
            self.crop = (self.rotate_to_image(self.r_selector.center), width, height)
        return self.crop

    def onclick(self, event):
        click_x = event.xdata
        click_y = event.ydata
        if event.button == 3:
            new_rot = (180 - degrees(atan2(click_x-self.centre[0],click_y-self.centre[1])))
            self.rotation = (self.rotation + new_rot) % 360
            ##recalibrate the crop coordinated
            #self.rect_callback(None, None)
        elif event.dblclick:
            self.zoom_point = self.rotate_to_image((click_x,click_y), self.centre)
        else:
            return
        self.update_fig()
        return

    def reset_fig(self):
        self.rotation = 0
        self.zoom_point = (0.5,0.5)
        self.crop = None
        self.crop_bounds = None
        self.mirror = False
        # self.update_fig()

    def clear_rect(self):
        self.crop = None
        self.crop_bounds = None
        self.update_fig()

    def get_data(self):
        self.get_rect()
        if self.crop:
            crop_data = (self.point_to_relative(self.crop[0]), self.crop[1], self.crop[2])
        else:
            crop_data = None
        return {"rotation":self.rotation, "mirror":self.mirror, 
                "zoom_point":self.point_to_relative(self.zoom_point), "crop":crop_data}

ndpa_template = """<?xml version="1.0" encoding="utf-8" standalone="yes"?>
<annotations>
	<ndpviewstate id="1">
		<title>zoom</title>
		<details/>
		<coordformat>nanometers</coordformat>
		<lens>1</lens>
		<x>0</x>
		<y>0</y>
		<z>0</z>
		<showtitle>1</showtitle>
		<showhistogram>0</showhistogram>
		<showlineprofile>0</showlineprofile>
		<annotation type="pin" displayname="AnnotatePin" color="#ff0000">
			<x>13077248</x>
			<y>-1711471</y>
			<icon>pinred</icon>
			<stricon>iconpinred</stricon>
		</annotation>
	</ndpviewstate>
	<ndpviewstate id="2">
		<title>crop</title>
		<details/>
		<coordformat>nanometers</coordformat>
		<lens>1</lens>
		<x>0</x>
		<y>0</y>
		<z>0</z>
		<showtitle>1</showtitle>
		<showhistogram>0</showhistogram>
		<showlineprofile>0</showlineprofile>
		<annotation type="freehand" displayname="AnnotateRectangle" color="#000000">
			<measuretype>0</measuretype>
			<closed>1</closed>
			<pointlist>
				<point>
					<x>9784406</x>
					<y>-4753072</y>
				</point>
				<point>
					<x>9784406</x>
					<y>843571</y>
				</point>
				<point>
					<x>18943280</x>
					<y>843571</y>
				</point>
				<point>
					<x>18943280</x>
					<y>-4753072</y>
				</point>
			</pointlist>
			<specialtype>rectangle</specialtype>
			<specialtype>rectangle</specialtype>
		</annotation>
	</ndpviewstate>
	<ndpviewstate id="3">
		<title>rotation</title>
		<details/>
		<coordformat>nanometers</coordformat>
		<lens>1</lens>
		<x>0</x>
		<y>0</y>
		<z>0</z>
		<showtitle>1</showtitle>
		<showhistogram>0</showhistogram>
		<showlineprofile>0</showlineprofile>
		<annotation type="linearmeasure" displayname="AnnotateRuler" color="#000000">
			<x1>10606418</x1>
			<y1>-537144</y1>
			<x2>18036854</x2>
			<y2>-537144</y2>
		</annotation>
	</ndpviewstate>
	<ndpviewstate id="4">
		<title>crop</title>
		<details />
		<coordformat>nanometers</coordformat>
		<lens>1</lens>
		<x>0</x>
		<y>0</y>
		<z>0</z>
		<showtitle>1</showtitle>
		<showhistogram>0</showhistogram>
		<showlineprofile>0</showlineprofile>
		<annotation type="pin" displayname="AnnotatePin" color="#ff0000">
			<x>4321</x>
			<y>-1711471</y>
			<icon>pinred</icon>
			<stricon>iconpinred</stricon>
		</annotation>
	</ndpviewstate>
</annotations>"""
