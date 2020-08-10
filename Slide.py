import numpy as np, openslide
from PIL import Image
import cv2
from skimage.color import rgb2hed
import xml.etree.cElementTree as ET
import os

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
            if annot.attrib['Type'] == "None":
                continue
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
        self.getWSI()

        if extraction_level > self.slide.level_count-1:
            self.extraction_level = self.slide.level_count-1
        else:
            self.extraction_level = extraction_level

        if annot_file == None or not os.path.exists(annot_file):
            self.annotation = None
        else:
            self.annotation = Annotation(annot_file, scaleFactor=self.slide.level_downsamples[self.extraction_level])

        self.mask_negatives = None
        self.mask_neighboring = None
        self.mask_annotations = None
        self.centroids = None
        

    def getWSI(self):
        '''
            Set OpenSlide from compatible format image filename
        '''
        self.slide = openslide.OpenSlide(self.slide_file)
        

    def getRegionFromSlide(self, start_coord=(0,0), dims='full', level=None):
        if level==None:
            level = self.extraction_level

        if dims == 'full':
            img = np.array(self.slide.read_region((0,0), level, self.slide.level_dimensions[level]))
            img = img[:,:,:3]
        else:
            img = np.array(self.slide.read_region(start_coord, level, dims))
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
        Return a scaled-channel image in the HED, color deconvolution performed format
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

    def getOtsu(self, img=None, level=None):
        if type(img)==type(None):
            img = self.getRegionFromSlide(level=level)
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img[img==0] = 255
        mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
        mask = self.performClose(self.performOpen(mask, kernel_size=5))

        return mask
        
    def getAnnotationMask(self):
        if self.annotation == None:
            return None
        dims = self.slide.level_dimensions[self.extraction_level]
        canvas = np.zeros((dims[1], dims[0]), dtype=np.uint8)

        mask = cv2.fillPoly(canvas, self.annotation.coords_list, 255)
        return mask

    def getAnnotationNeighborhoodMask(self, kernel_size_inland=5, kernel_size_coast=2):
        '''
        Return a mask image with non-zero values only around the region of the annotation mask.

        Simplistically, the function perform an XOR operation between the Annotation mask and its Dilated version. 
        Additional logical operation are performed to reduce the area in the blank spots (not containing specimen scan) - called coast and 
        increase the area in the informative region (containing specimen scan) - called inland. 
        '''
        
        kernel_inland = np.ones((kernel_size_inland, kernel_size_inland), np.uint8)
        kernel_coast = np.ones((kernel_size_coast, kernel_size_coast), np.uint8)
        otsu_mask = self.getOtsu()

        annotation_mask = self.getAnnotationMask()
        #annotation_mask_dilated = cv2.dilate(annotation_mask, kernel, iterations=1)

        #n_mask = np.logical_xor(annotation_mask, annotation_mask_dilated)

        annot_inland = np.logical_and(annotation_mask, otsu_mask)
        annot_coast = np.logical_and(annotation_mask, np.logical_not(otsu_mask))

        annot_inland_dilated = cv2.dilate(annot_inland.astype(np.uint8)*255, kernel_inland)
        annot_coast_dilated = cv2.dilate(annot_coast.astype(np.uint8)*255, kernel_coast)

        annotation_mask_dilated = np.logical_or(annot_inland_dilated, annot_coast_dilated)

        n_mask = np.logical_xor(annotation_mask, annotation_mask_dilated)
        #n_mask_inland = np.logical_xor(annot_inland, annot_inland_dilated)
        #n_mask_coast = np.logical_xor(annot_coast, annot_coast_dilated)

        #n_mask = np.logical_or(n_mask_inland, n_mask_coast)
        
        n_mask = 255*(n_mask.astype(np.uint8))

        return n_mask
    
    def getDABMask(self, img=None, margins=(0, 0, 0, 0), level=None):
        '''
        Returns the bit mask for ROI from the given RGB img, using DAB channel of the HED (deconvoluted) image
        '''
        if img==None:
            img = self.getRegionFromSlide(level=level)
        
        hed = self.getHED(img)
        mask = self.getDABThresholdMask(hed, margins=margins)
        mask = self.performOpen(self.performClose(mask))

        return mask

    def getGTmask(self, coords, dims=(256,256), level=1, fill_val=1):
        if dims=='full':
            dims = self.slide.level_dimensions[level]

        if self.annotation==None:
            return np.zeros((dims[1], dims[0], 1), dtype=np.uint8)

        c_shifted = self.annotation.shift(origin=coords)
        c_scaled = self.annotation.scale(c_shifted, self.slide.level_downsamples[level])

        mask = cv2.fillPoly(np.zeros((dims[1], dims[0], 1), dtype=np.uint8), c_scaled, (fill_val))

        return mask

    def getGTMaskedRegion(self, coords, dims=(256,256), level=1, withMaskArea=False):
        if dims=='full':
            dims = self.slide.level_dimensions[level]
        img = self.getRegionFromSlide(coords, dims, level)
        mask = self.getGTmask(coords, dims, level)
        mask_3 = np.dstack((np.zeros_like(mask), mask*255, np.zeros_like(mask)))


        if withMaskArea:
            return cv2.addWeighted(img, 0.8, mask_3, 0.4, 0), np.sum(mask)
        else:
            return cv2.addWeighted(img, 0.8, mask_3, 0.4, 0)

    def getLabel(self, coords, dims=(256,256), level=1):
        '''
        Return [1.0, 0.0] if the region at the specified rectangulare area (dims) starting at given coordinates (coords) does not have a detection else, 
        return [0.0, 1.0] if the region at the specified rectangulare area (dims) starting at given coordinates (coords) has a detection 
        '''
        if self.annotation==None:
            return np.array([1.0, 0.0])
        else:
            detection = np.any(self.getGTmask(coords, dims, level))
            label = np.array( [float(not detection), float(detection)] )
            return label

    def generateROIMasks(self, thresh_method='HED', skip_negatives=False):
        '''
        Calculate and store as object variables: 
            *a mask of the area containing regions of interest in the slide (containing the specimen scan) while exlcuding any regions of positive detection and areas around it
            *a mask of the annotated area (if available)
            *a mask of the area surrounding the annotations 

        The ROI selection is done based on the given thresh_method, which could be either 
                'HED' (DAB channel of the color deconvoluted, rescaled image) or
                'OTSU' (Otsu thresholded mask)
        '''
        if thresh_method == 'HED':
            mask_whole = self.getDABMask()
        elif thresh_method == 'OTSU':
            mask_whole = self.getOtsu()

        self.mask_whole = mask_whole

        if self.annotation == None:
            self.mask_negatives = mask_whole
            return

        self.mask_annotations = self.getAnnotationMask()
        self.mask_neighboring = self.getAnnotationNeighborhoodMask()

        if skip_negatives:
            return
        
        self.mask_negatives = np.logical_and(mask_whole, np.logical_not(np.logical_or(self.mask_annotations, self.mask_neighboring)))
        self.mask_negatives = self.mask_negatives.astype(np.uint8)*255


    def getNonZeroLocations(self, mask, with_filename=True):
        nzs = np.argwhere(mask)
        nzs = nzs * self.slide.level_downsamples[self.extraction_level]
        nzs = nzs.astype(np.int32)
        nzs = np.flip(nzs, 1)

        if not with_filename:
            return nzs

        l = []
        for i in range(nzs.shape[0]):
            l.append([self.slide_file, nzs[i].tolist()])

        return l


    def getPatchCoordList(self, thresh_method='OTSU', with_filename=True, skip_negatives=False):
        '''
        Returns a list of coordinates in the slide image where useful data is expected to be present by using 
        thresholding to remove blank or non-informative areas, at the given image level (extraction_level)

        This function gets a downsampled version of the slide image at level either the highest level or the one specified in self.extraction_level
        Then, it applies a thresholding technique which could be either 
                'HED' (DAB channel of the color deconvoluted, rescaled image) or
                'OTSU' (Otsu thresholded mask)
        depending on the 'thresh_method' value.
        Following the mask generation, the indices of all values that are non-zero are recorded (with_filename if true) and returned as a list  
        '''

        self.generateROIMasks(thresh_method=thresh_method, skip_negatives=skip_negatives)
        
        if skip_negatives:
            if self.annotation==None:
                return []
            else:
                return [[], \
                    self.getNonZeroLocations(self.mask_annotations, with_filename), \
                    self.getNonZeroLocations(self.mask_neighboring, with_filename)]


        if self.annotation==None:
            return [self.getNonZeroLocations(self.mask_negatives)]
        else:
            return [self.getNonZeroLocations(self.mask_negatives, with_filename), \
                self.getNonZeroLocations(self.mask_annotations, with_filename), \
                self.getNonZeroLocations(self.mask_neighboring, with_filename)]

    def getPatchCoordListWLabels(self, thresh_method='OTSU', with_filename=True, view_level=1, skip_negatives=False):
        if not skip_negatives and self.annotation==None:
            patch_coords_list = self.getPatchCoordList(thresh_method, with_filename)
            assert len(patch_coords_list)==1
            patch_coords_list = patch_coords_list[0]
            return_list = []
            for item in patch_coords_list:
                return_list.append([item[0], item[1], self.getLabel(item[1], level=view_level).tolist()])
            return [return_list]

        else:
            patch_coords_lists = self.getPatchCoordList(thresh_method, with_filename, skip_negatives=skip_negatives)
            return_lists = []
            for patch_coords_list in patch_coords_lists:
                return_list = []
                for item in patch_coords_list:
                    return_list.append([item[0], item[1], self.getLabel(item[1], level=view_level).tolist()])
                return_lists.append(return_list)
            return return_lists

    def getTileListWLabels(self, thresh_method='OTSU', view_level=1):
        '''
        get Tile and correponding labels list for validation runs - returns only one list (no annotation or neighboring lists)
        '''        
        self.generateROIMasks(thresh_method=thresh_method)
        patch_coords_list = self.getNonZeroLocations(self.mask_whole, with_filename=False)
        label_list = []
        for item in patch_coords_list:
            label_list.append(self.getLabel(item, level=view_level).tolist())
        return patch_coords_list, label_list

        
    def getCentroids(self, level=0, extraction_level=4):
        
        if self.annotation!=None:
            if self.centroids == None:
                d = self.slide.level_dimensions[extraction_level]
                downsample = self.slide.level_downsamples[extraction_level]
                canvas = np.zeros((d[1], d[0], 1), dtype=np.uint8)
                centroids = []
                c_scaled = self.annotation.scale(factor=downsample)
                partOfGroup = self.annotation.group
                for c,grp in zip(c_scaled, partOfGroup):
                    if grp == "normal": continue
                    cv2.fillPoly(canvas, [c], 255)
                    contour = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    try:
                        M = cv2.moments(contour[1][0])
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                        centroids.append([cx,cy])
                    except:
                        print(c*downsample)                 
                   
                    canvas = canvas*0
                if "normal" in partOfGroup:
                    print("found normal annotation in file: ", self.slide_file)
                    print("# of centroids/annotation: ", len(centroids), "/", len(c_scaled))
                else:
                    assert len(centroids) == len(c_scaled)
                centroids = np.array(centroids)
                self.centroids = (centroids * downsample).astype(np.uint)

            return (self.centroids / self.slide.level_downsamples[level]).astype(np.uint)
        else:
            return None

    def getNonblankBB(self, img=None, level=None, thresh_method='OTSU'):
        if level==None:
            level = self.extraction_level
        
        if thresh_method=="OTSU":
            mask = self.getOtsu(img=img, level=level)
        elif thresh_method=="HED":
            mask = self.getDABMask(img=img,level=level)
        else:
            return None
        
        nz = np.argwhere(mask)
        XYmin = (np.min(nz[:, 1]), np.min(nz[:, 0]))
        XYmax = (np.max(nz[:, 1]), np.max(nz[:, 0]))

        return np.array([XYmin, XYmax])
        