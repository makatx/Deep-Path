import numpy as np, openslide
from PIL import Image
import cv2
from skimage.color import rgb2hed
import xml.etree.cElementTree as ET

class Annotation:
    '''
    Objects of this class represent and store all markup/annotation info from a single ASAP style XML file
    developed as part of the project here: https://computationalpathologygroup.github.io/ASAP/
    
    Annotation coordinate groups are stored with original scale in coords_orig and scaled as per scaleFactor in coord_list
    The class functions can also be called on given coordinate list to scale and shift according to give scale factor and new origin repectively

    Finally the class has calcBounds() function that runs on member variables to generate bounding box coordinates for each annotation in the file
    and saves them in bounds (scaled) and bounds_orig member variables
    
    '''
    
    def __init__(self, filename, scaleFactor=1):
        self.scaleFactor = scaleFactor
        with open(filename, 'rb') as f:
            self.root = ET.parse(f)
        self.coords_orig = []
        self.coords_order = []
        self.group = []
        self.type = []
        
        for annot in self.root.iter('Annotation'):
            coords_tag = annot.find('Coordinates')
            lst = []
            for coord in coords_tag.findall('Coordinate'):
                lst.append([float(coord.attrib['Order']), float(coord.attrib['X']), float(coord.attrib['Y'])])
            n = np.array(lst)
            n = n[n[:,0].argsort()]
            self.coords_orig.append(n[:,1:])
            self.coords_order.append(n)
            self.group.append(annot.attrib['PartOfGroup'])
            self.type.append(annot.attrib['Type'])
        
        self.coords_list = self.scale(factor=scaleFactor)
        self.calcBounds()
            
    def scale(self, coords=None, factor=1):
        if coords == None: coords = self.coords_orig
        coords_scaled = []
        for n in range(len(coords)):
            coords_scaled.append((coords[n] / factor).astype(np.int))
        return coords_scaled
    
    def shift(self, coords=None, origin=(0,0)):
        if coords == None: coords = self.coords_orig
        shifted = []
        origin = np.array(origin)
        for n in coords:
            shifted.append(n - origin)
        return shifted
    
    def calcBounds(self):
        bounds = []
        for n in self.coords_list:
            xmin = n[:,0].min()
            ymin = n[:,1].min()
            xmax = n[:,0].max()
            ymax = n[:,1].max()
            bounds.append(np.array([xmin,ymin,xmax,ymax]))
        self.bounds = np.array(bounds)
        bounds = []
        for n in self.coords_orig:
            xmin = n[:,0].min()
            ymin = n[:,1].min()
            xmax = n[:,0].max()
            ymax = n[:,1].max()
            bounds.append(np.array([xmin,ymin,xmax,ymax]))
        self.bounds_orig = np.array(bounds)
    
class Slide:

    def __init__(self, slide_file, annot_file=None, extraction_level=7):
        self.slide_file = slide_file
        self.annot_file = annot_file
        self.extraction_level = extraction_level

        if annot_file == None:
            self.annotation = None
        else:
            self.annotation = Annotation(annot_file)

        self.getWSI()
        

    def getWSI(self):
        '''
            Set OpenSlide from compatible format image filename
        '''
        self.slide = openslide.OpenSlide(self.slide_file)
        

    def getRegionFromSlide(self, start_coord=(0,0), dims='full'):
        if dims == 'full':
            img = np.array(self.slide.read_region((0,0), self.extraction_level, self.slide.level_dimensions[self.extraction_level]))
            img = img[:,:,:3]
        else:
            img = np.array(self.slide.read_region(start_coord, self.extraction_level, dims))
            img = img[:,:,:3]
        
        return img 

    def getThresholdMask(self, img, threshold=(140,210), channel=0, margins=None):
        '''
        Retuns threhold applied image for given threhold and channel, suppressing any pixels to 0 for given margins

        params:
        margins: (left_y, right_y, top_x, bottom_x) ;  can be specified as negative as well. ex: (50, -50, 50, -50)

        '''
        mask = np.zeros_like(img[:,:,channel], dtype=np.uint8)
        mask[((img[:,:,channel] > threshold[0]) & (img[:,:,channel] < threshold[1]))] = 255

        if margins != None :
            mask[:margins[0]] = 0
            mask[margins[1]:] = 0

            mask[:, :margins[2]] = 0
            mask[:, margins[3]:] = 0

        return mask


    def getDABThresholdMask(self, hed_img, threshold=(30,150), margins=(50, -50, 50, -50)):
        return self.getThresholdMask(hed_img, threshold, channel=2, margins=margins)


    def getHED(self, img):
        '''
        Return a channel scaled image in the HED, color deconvolution performed format
        '''
        hed = rgb2hed(img)
        #hed_sc = np.zeros_like(hed)
        for i in range(3):
            r_min = np.min(hed[:,:,i])
            r_max = np.max(hed[:,:,i])
            r = r_max - r_min
            hed[:,:,i] = (hed[:,:,i]-r_min) * 255.0/r
        return hed.astype(np.uint8)

    def performClose(self, img, kernel_size=3):
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    def performOpen(self, img, kernel_size=3):
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    def getOtsu(self, img=None):
        if img==None:
            img = self.getRegionFromSlide()
        
        return cv2.threhold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        

    def getDABMask(self, img=None, margins=(25, -25, 25, -35)):
        '''
        Returns the bit mask for ROI from the given RGB img, using DAB channel of the HED converted image
        '''
        if img==None:
            img = self.getRegionFromSlide()
        
        hed = self.getHED(img)
        mask = self.getDABThresholdMask(hed, margins=margins)
        mask = self.performOpen(self.performClose(mask))

        return mask

    def getPatchCoordList(self, thresh_method='HED', from_level=None, with_filename=True):
        #if from_level==None :
        #    from_level = self.extraction_level
        #elif from_level='max' or  from_level > self.slide.level_count-1:
        #    from_level = self.slide.level_count-1


        #img = self.getRegionFromSlide()
        
        ##TODO: Add Otsu thresholding method as option to extract point of interest
        
        if thresh_method == 'HED':
            mask = self.getDABMask()
        else:
            mask = self.getOtsu()

        nzs = np.argwhere(mask)
        nzs = nzs * self.slide.level_downsamples[from_level]
        nzs = nzs.astype(np.int32)
        nzs = np.flip(nzs, 1)

        if not with_filename:
            return nzs

        l = []
        for i in range(nzs.shape[0]):
            l.append([filename, nzs[i].tolist()])

        return l
