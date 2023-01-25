from math import degrees, atan2, radians, cos, sin
from PIL import Image, ImageDraw, ImageOps
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
# Windows: this must be overwritten with the path to openslide\bin before first slide objust creation
OPENSLIDE_PATH = None

# Image manipulation functions with mpp scale retention
def fit_image(img, new_size, method = Image.Resampling.LANCZOS):
    """A wrapper for PIL thumbnail() which returns a padded image 
    of required size with mpp scale information retention.
    img: image to be fit
    new_size: tuple pixel size for image returned
    method: PIL.Image resampling method passed to PIL thumbnail(). Default Image.Resampling.LANCZOS
    returns: downscaled PIL image with mpp scale property recalculated
    """
    w1, h1 = img.size
    if new_size[0] > w1 and new_size[1] > h1:
        msg = "Thumbnail: new_size > current size"
        raise ValueError(msg)
    img.thumbnail(new_size, method)
    w2, h2 = img.size
    downscale = max(w1/w2, h1/h2)
    img2 = Image.new('RGBA', new_size, "#ffffff")
    img2.paste(img, ((new_size[0]-w2)//2,(new_size[1]-h2)//2), None)      
    if hasattr(img,"mpp"):
        img2.mpp = img.mpp*downscale
    return img2

def thumb_image(img, new_size, method = Image.Resampling.LANCZOS):
    """A wrapper for PIL thumbnail() which returns a thumbnailed image 
    of maximum new_size with mpp scale information retention.
    img: image to be fit
    new_size: tuple pixel size for max dimensions of image returned
    method: PIL.Image resampling method passed to PIL thumbnail(). Default Image.Resampling.LANCZOS
    returns: downscaled PIL image with mpp scale property recalculated
    """
    w1, h1 = img.size
    if new_size[0] > w1 and new_size[1] > h1:
        msg = "Thumbnail: new_size > current size"
        raise ValueError(msg)
    img.thumbnail(new_size, method)
    w2, h2 = img.size
    downscale = max(w1/w2, h1/h2)
    if hasattr(img,"mpp"):
        img.mpp = img.mpp*downscale
    return img

def add_scalebar(img, sb = 100, ratio = 50, mpp = None, color = "#000000", pad = 3):
    """Add a scalebar to image file.
    Scale information is supplied or taken from image.mpp.
    Scale information in retained in returned image.
    img: PIL image to add scale bar to.
    sb: length in um of scalebar. default 100um
    ratio: height of scalebar relative to image height. default 50
    mpp: microns per pixel scale to use. If not supplied, img.mpp will be used.
    color: color of scalebar
    pad: distance of scalebar from image edge
    returns: PIL image with mpp data retained"""
    if mpp is not None:
        img.mpp = mpp
    elif hasattr(img,"mpp"):
        mpp = img.mpp
    else:
        msg = "No mpp data."
        raise ValueError(msg)
    draw = ImageDraw.Draw(img)
    #number of pixels in scalebar
    sb_px_x = sb//mpp
    sb_px_y = max(img.height//ratio,5) #ratio scale bar to image height
    #draw scalebar
    draw.rectangle(((0+pad, img.height-sb_px_y-pad), (sb_px_x+pad, img.height-pad)), fill=color, outline=color)
    return img

def apply_border(img, bw = 2, color = "#000000"):
    """Applies a border to supplied image by thumbnailing image 
    on a black background. Image mpp scale information is adjusted.
    img: image to apply border to
    bw: border width (pixels) default 2
    color: border color
    returns: image with PIL mpp data retailed"""
    w1, h1 = img.size
    img = fit_image(img, (w1-2*bw,h1-2*bw))
    w2, h2 = img.size
    downscale = max(w1/w2, h1/h2)
    img2 = Image.new('RGBA', (w1,h1), color)
    img2.paste(img, (bw,bw), None)
    if hasattr(img,"mpp"):
        img2.mpp = img.mpp*downscale
    return img2

# class wrapper for openslide object with added picture output formatting functions
class slide_obj:
    openslide_imported = False
    
    def __init__(self, file, rotation = 0, mirror = False, zoom = (0.5,0.5), crop = ((0.5,0.5),1,1)):
        if not slide_obj.openslide_imported:
            global openslide
            if hasattr(os, 'add_dll_directory'):
                # Python >= 3.8 on Windows
                if OPENSLIDE_PATH is None or OPENSLIDE_PATH == "":
                    msg = "Please specify path to OpenSlide\\bin"
                    raise ModuleNotFoundError(msg)
                with os.add_dll_directory(Path(OPENSLIDE_PATH)):
                    import openslide
            else:
                import openslide
            slide_obj.openslide_imported = True
        self.filename = Path(file)
        self.slide = openslide.OpenSlide(self.filename)
        self.mpp_x = float(self.slide.properties["openslide.mpp-x"])
        self.rot = float(rotation) if pd.notnull(rotation) else 0
        self.mirror = mirror if pd.notnull(mirror) else False
        self.crop = eval(str(crop)) if pd.notnull(crop) else ((0.5,0.5),1,1)
        self.zoom_point = eval(str(zoom)) if pd.notnull(zoom) else (0.5,0.5)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exception_type, exception_value, exception_traceback) :
        self.close()
        
    def close(self):
        self.slide.close()
        self.slide = None
        
    def get_raw_overview_image(self, image_size = (500,500), sb=None):
        """ .get_raw_overview_image(image_size = (500,500), sb=None)
            Return a PIL.Image containing an RGBA thumbnail of the slide.
            Rotations and crop are not applied.
            size:     overall dimensions of the image.
            sb:       size of scalebar in um.
        """
        downsample = max(*(dim / thumb for dim, thumb in zip(self.slide.dimensions, image_size)))
        level = self.slide.get_best_level_for_downsample(downsample)
        tile = self.slide.read_region((0, 0), level, self.slide.level_dimensions[level])

        img = Image.new('RGBA', tile.size, "#ffffff")
        img.paste(tile, None, None)
        draw = ImageDraw.Draw(img)

        #calculate um per pixel at this downsample
        mpp_x = self.mpp_x * self.slide.level_downsamples[level]
        img.mpp = mpp_x
        img = thumb_image(img, image_size)
        if sb:
            img = add_scalebar(img, sb=sb)        
        return img

    def _get_sub_image(self, image_size, image_centre = (0.5,0.5), 
                      true_size = None, rot = None, mirror = None, sb = None):
        image_width = image_size[0] #300 #px
        image_height = image_size[1]
        #image_size = (image_width, image_width)
        if true_size is None:
            true_size = tuple(i * self.mpp_x for i in self.slide.dimensions)
        elif len(true_size) == 1:
            true_size = (true_size[0],true_size[0]*(image_height/image_width))
        true_width = true_size[0]
        true_height = true_size[1]

        target_mpp = true_width/image_width
        target_downsample = target_mpp / self.mpp_x
        level = self.slide.get_best_level_for_downsample(target_downsample)
        level_downsample = self.slide.level_downsamples[level]

        level_mpp = level_downsample * self.mpp_x
        
        #get tile of double largest side centred on zoom_point, later crop out centre
        tile_width = int(max(true_width, true_height)*2/level_mpp)
        tile_x = int((self.slide.dimensions[0]*image_centre[0]) - tile_width*level_downsample/2)
        tile_y = int((self.slide.dimensions[1]*image_centre[1]) - tile_width*level_downsample/2)
        
        tile = self.slide.read_region((tile_x,tile_y), level, (tile_width,tile_width))
                
        img = Image.new('RGBA', tile.size, "#ffffff")
        img.paste(tile, (0,0), None)
        
        if mirror:
            img = ImageOps.mirror(img)
            
        if rot:
            img = img.rotate(rot, fillcolor="#ffffff", expand=False)
        
        #crop image to required dimensions
        crop_width = true_width/level_mpp
        crop_height = true_height/level_mpp
        w, h = img.size
        crop_x = (w-crop_width)/2
        crop_y = (h-crop_height)/2
        img = img.crop((crop_x, crop_y, w-crop_x, h-crop_y))
        
        img.mpp = level_mpp
        img = fit_image(img, image_size)
        if sb:
            img = add_scalebar(img, sb=sb)
        return img
    
    def get_zoom_image(self, image_size, true_width, zoom_point = None, 
                       rot = None, mirror = None, sb = None):
        """ .get_zoom_image(image_size, true_width, zoom_point = None, 
                       rot = None, mirror = None, sb = None)
            returns a PIL.Image 
        """
        if zoom_point is None:
            zoom_point = self.zoom_point
        if rot is None:
            rot = self.rot
        if mirror is None:
            mirror = self.mirror
        return self._get_sub_image(image_size, image_centre = zoom_point, 
                                  true_size = (true_width,), rot = rot, mirror = mirror, sb = sb)
    
    def get_crop_image(self, image_size, rot = None, mirror = None, crop = None, sb = None):
        """ .get_crop_image(image_size, rot = None, mirror = None, 
                            crop = None, sb = None)
            returns PIL.Image of cropped and rotated ROI from slide
            required:
            image_size
            if other parameters are None, the slide_obj properties are used.
        """
        if rot is None:
            rot = self.rot
        if mirror is None:
            mirror = self.mirror
        if crop is None:
            crop = self.crop
        return self._get_sub_image(image_size, image_centre = crop[0], 
                                  true_size = self._relative_to_true((crop[1],crop[2])),
                                  rot = rot, mirror = mirror, sb = sb)
    
    def get_overview_image(self, image_size, rot = None, mirror = None, sb = None):
        """ .get_overview_image(image_size, rot = None, mirror = None, sb = None)
            returns PIL.Image of whole slide with rotation and mirroring
            required:
            image_size
            if other parameters are None, the slide_obj properties are used.
        """
        if rot is None:
            rot = self.rot
        if mirror is None:
            mirror = self.mirror
        return self._get_sub_image(image_size,
                                  rot = rot, mirror = mirror, sb = sb)
    
    def _relative_to_l0(self, rel):
        return (rel[0]*self.slide.dimensions[0], rel[1]*self.slide.dimensions[1])
    
    def _relative_to_true(self, rel):
        return (rel[0]*self.slide.dimensions[0]*self.mpp_x, rel[1]*self.slide.dimensions[1]*self.mpp_x)

    def _rotate_point(self, loc1, rot = 0, centre = (0,0)):
        rads = radians(rot)
        new_x = centre[0] + cos(rads) * (loc1[0] - centre[0]) - sin(rads) * (loc1[1] - centre[1])
        new_y = centre[1] + sin(rads) * (loc1[0] - centre[0]) + cos(rads) * (loc1[1] - centre[1])
        #loc2 = (new_x/image1.size[0], new_y/image1.size[1])
        return (new_x, new_y)
    
    def get_figure(self, image_size = (500,500), add_inset = True, inset_size = None, inset_true = 250, 
                   rot = None, mirror = None, zoom_point = None, crop = None, sb = False, inset_sb = False):
        """ .get_figure(self, image_size = (500,500), add_inset = True, 
                    inset_size = None, inset_true = 250, rot = None, 
                    mirror = None, zoom_point = None, crop = None, sb = False, 
                    inset_sb = False)
            returns PIL.Image containing cropped overview image with inset zoom image (add_inset = True).
            If parameters are None, slide_obj parameters are used.
        """
        if rot is None:
            rot = self.rot
        if mirror is None:
            mirror = self.mirror
        if crop is None:
            crop = self.crop
        if zoom_point is None:
            zoom_point = self.zoom_point
        if inset_size is None:
            inset_size = (image_size[0]//3,image_size[0]//3)
        
        base_image = apply_border(self.get_crop_image(
                                                image_size, rot = rot, mirror = mirror, 
                                                crop = crop, sb = sb))
        if add_inset:
            zoomed_image = apply_border(self.get_zoom_image(
                                                inset_size, inset_true, zoom_point = zoom_point, 
                                                rot = rot, mirror = mirror, sb = inset_sb))
            base_image.paste(zoomed_image, (int(base_image.width-zoomed_image.width),0))
        
            relative_zoom_point = tuple(i-j for i, j in zip(zoom_point,crop[0]))
            if mirror:
                relative_zoom_point = (-relative_zoom_point[0],relative_zoom_point[1])
            # calculate um distance between middle of crop and zoom point on non-rotated image as is relative to non-rotated scan dimensions
            zoom_point_offset = (relative_zoom_point[0]*self.slide.dimensions[0]*self.mpp_x,
                                 relative_zoom_point[1]*self.slide.dimensions[1]*self.mpp_x)
            zoom_point_offset = self._rotate_point(zoom_point_offset, -rot, (0,0))

            box_x = zoom_point_offset[0] / base_image.mpp + 0.5*base_image.width
            box_y = zoom_point_offset[1] / base_image.mpp + 0.5*base_image.height
            box_width = inset_true/base_image.mpp

            draw = ImageDraw.Draw(base_image)
            draw.rectangle((
                (int((box_x+box_width/2)),
                 int((box_y-box_width/2))),
                (int((box_x-box_width/2)),
                 int((box_y+box_width/2)))
                ), outline="black")

        return base_image
# end slide_obj class


class pathofigure:
    # Variables to set page layout
    panel_size = (500,500)
    scale_bar = 1000
    scale_bar_mm = scale_bar / 1000 #for footer

    add_inset = True
    inset_size = (200,200)
    inset_real_size = 250
    inset_scale_bar = None

    figsize = (8.27,11.69) #A4 in inches
    n_panels_x = 4
    n_panels_y = 6

    # str.format(**globals()) is applied at time of use.
    title_overwrite = None
    footer_overwrite = "Scale bar {pathofigure.scale_bar_mm}mm. Inset image width {pathofigure.inset_real_size}um."

    @staticmethod
    def fig_from_df(df, page_row):#title = None, footer = None):
        if len(df) > (pathofigure.n_panels_y * pathofigure.n_panels_x):
            msg = "Too many panels for layout!"
            raise ValueError(msg)

        fig, axs = plt.subplots(pathofigure.n_panels_y, pathofigure.n_panels_x, 
                                constrained_layout=True, figsize=pathofigure.figsize, dpi=300)
        for ax in axs.ravel():
            ax.axis('off')

        if pathofigure.title_overwrite is not None:
            title = pathofigure.title_overwrite.format(**globals())
        else:
            title_list = [str(t) for t in 
                          [page_row["title1"],page_row["title2"],page_row["title3"]] if pd.notnull(t)]
            title = "\n".join(title_list).format(**globals())

        if pathofigure.footer_overwrite is not None:
            footer = pathofigure.footer_overwrite.format(**globals())
        elif pd.notnull(page_row["footer"]):
            footer = page_row["footer"].format(**globals())
        else:
            footer = ""

        fig.suptitle(title)
        fig.supxlabel(footer)

        for index, row in df.iterrows():
            ax = axs.ravel()[row["order"]]
            if pd.notnull(row["label"]):
                ax.set_title(row["label"], y=0, loc="right", pad=2)

            if pd.notnull(row['file']):
                with slide_obj(Path(row['root'],row['file'])) as sld:
                    if pd.notnull(row["rotation"]): sld.rot = float(row["rotation"])
                    if pd.notnull(row["mirror"]): sld.mirror = row["mirror"]
                    # excel import will be str, direct from interactive will be tuple. Force to str then eval to tuple
                    if pd.notnull(row["crop"]): sld.crop = eval(str(row["crop"])) 
                    if pd.notnull(row["zoom_point"]): sld.zoom_point = eval(str(row["zoom_point"]))
                    image = sld.get_figure(
                        pathofigure.panel_size,
                        add_inset = pathofigure.add_inset,
                        inset_size = pathofigure.inset_size, 
                        inset_true = pathofigure.inset_real_size, 
                        sb = pathofigure.scale_bar,
                        inset_sb = pathofigure.inset_scale_bar)
                    ax.imshow(image)
                    image.close()
        return fig

    pdf_title = "Histopathology summary images"
    pdf_author = "Created with PATHOverview: github.com/EpiCENTR-Lab/PATHOverview"
    pdf_subject = None
    pdf_keywords = "Created with PATHOverview: github.com/EpiCENTR-Lab/PATHOverview"
    pdf_creationdate = str(datetime.date.today())

    @staticmethod
    def save_pdf(figs, filename):
        with PdfPages(Path(filename)) as pdf:
            for f in figs:
                pdf.savefig(f)
            d = pdf.infodict()
            d['Title'] = pathofigure.pdf_title
            d['Author'] = pathofigure.pdf_author
            d['Subject'] = pathofigure.pdf_subject
            d['Keywords'] = pathofigure.pdf_keywords
            d['CreationDate'] = pathofigure.pdf_creationdate
# end pathofigure

class pathoverview_interactive_fig:
    def __init__(self, filename, rot = 0, mirror = False, zoom = (0,0), crop = None):
        with plt.ioff():
            self.fig = plt.figure(figsize=(8,8))
        self.fig.canvas.toolbar_visible = False
        self.ax = self.fig.add_subplot()#1, 1, 1)
        self.load_image(filename, rot, mirror, zoom, crop)
        self.draw_fig()
        self.click_listen = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self):
        self.fig.close()

    def load_image(self, filename, rot = 0, mirror = False, zoom = (0,0), crop = None):
        with slide_obj(Path(filename)).get_raw_overview_image() as img:
            self.image = img
        self.rot = rot
        self.expand_rot = True
        self.mirror = mirror
        # middle of the zoom image relative to the center of image1
        self.zoom_point = zoom
        self.crop = crop
        self.crop_bounds = None
        self.width = self.image.width
        self.height = self.image.height
        self.zoom_dot = None
        self.centre = (0,0)
    
    def load_data(self,row):
        if pd.notnull(row["rotation"]): self.rot = float(row["rotation"])
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
        self.update_fig()

    # @output.capture()
    def draw_fig(self):
        image2 = self.image
        if self.mirror:
            image2 = ImageOps.mirror(image2)
        image2 = image2.rotate(self.rot, fillcolor="#ffffff", expand=self.expand_rot)
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
        self.zoom_dot = self.ax.scatter(zoom_loc[0], zoom_loc[1], color="g")
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
        rads = -radians(self.rot)
        new_x = centre[0] + cos(rads) * (loc1[0] - centre[0]) - sin(rads) * (loc1[1] - centre[1])
        new_y = centre[1] + sin(rads) * (loc1[0] - centre[0]) + cos(rads) * (loc1[1] - centre[1])
        return (new_x, new_y)
    
    def rotate_to_image(self, loc1, centre = (0,0)):
        rads = radians(self.rot)
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
            self.rot = (self.rot + new_rot) % 360
            ##recalibrate the crop coordinated
            #self.rect_callback(None, None)
        elif event.dblclick:
            self.zoom_point = self.rotate_to_image((click_x,click_y), self.centre)
        else:
            return
        self.update_fig()
        return

    def reset_fig(self):
        self.rot = 0
        self.zoom_point = (0.5,0.5)
        self.crop = None
        self.crop_bounds = None
        self.mirror = False
        self.update_fig()

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
        return {"rotation":self.rot, "mirror":self.mirror, 
                "zoom_point":self.point_to_relative(self.zoom_point), "crop":crop_data}
