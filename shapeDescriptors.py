import numpy as np
import cv2
import matplotlib.pyplot as plt
from copy import deepcopy as dc
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
import math
import pyrealsense2.pyrealsense2 as rs



image = cv2.imread("bagPNGs/tableFruits.png")
rgb_masks = np.load("randomItems_rgb_masks.npy", allow_pickle=True)
#depth_masks = np.load("randomItems_depth_masks.npy", allow_pickle=True)


black_thresh = 50
white_thresh = 50
decay_rate = 20
default_s = 255
light_s = 80
default_v = 225
dark_v = 100
colorsHSV = {  "Black": (0, 0, 0),
            "White": (0, 0, 255),
            "Gray": (0, 0, 128),
            "Red": (0, default_s, default_v),
            "LightRed": (0, light_s, default_v),
            "DarkRed": (0, default_s, dark_v),
            "Orange": (15, default_s, default_v),
            "LightOrange": (15, light_s, default_v),
            "DarkOrange": (15, default_s, dark_v),
            "Yellow": (30, default_s, default_v),
            "LightYellow": (30, light_s, default_v),
            "DarkYellow": (30, default_s, dark_v),
            "Green": (60, default_s, default_v),
            "LightGreen": (60, light_s, default_v),
            "DarkGreen": (60, default_s, dark_v),
            "Cyan": (90, default_s, default_v),
            "LightCyan": (90, light_s, default_v),
            "DarkCyan": (90, default_s, dark_v),
            "Blue": (115, default_s, default_v),
            "LightBlue": (115, light_s, default_v),
            "DarkBlue": (115, default_s, dark_v),
            "Purple": (130, default_s, default_v),
            "LightPurple": (130, light_s, default_v),
            "DarkPurple": (130, default_s, dark_v),
            "Pink": (150, default_s, default_v),
            "LightPink": (150, light_s, default_v),
            "DarkPink": (150, default_s, dark_v),
        }

colors_image = np.zeros((512, 512, 3), dtype=np.uint8)
color_width = colors_image.shape[1] // len(colorsHSV.keys())
for i, color in enumerate(colorsHSV.keys()):
    colors_image[:, i*color_width:(i+1)*color_width] = colorsHSV[color]


def activationFunction(x):
    return 1 / (np.power(2, x/decay_rate))
def pixelColorScores(pixelHSV):
    scores = {}
    for color in colorsHSV.keys():
        scores[color] = 0
        if color == "Black":
            dist = np.abs(pixelHSV[2]-colorsHSV[color][2])
            scores[color] += 1 / (np.power(2, dist/8))
        elif color == "White":
            dist = np.sqrt(0.6*(pixelHSV[2]-colorsHSV[color][2])**2 + 0.4*(pixelHSV[1]-colorsHSV[color][1])**2)
            scores[color] += 1 / (np.power(2, dist/8))
        elif color == "Gray":
            dist = np.sqrt(0.6*(pixelHSV[2]-colorsHSV[color][2])**2 + 0.4*(pixelHSV[1]-colorsHSV[color][1])**2)
            scores[color] += 1 / (np.power(2, dist/8))
        else:
            dist = np.sqrt(0.4*(pixelHSV[0]-colorsHSV[color][0])**2 + 0.3*(pixelHSV[1]-colorsHSV[color][1])**2 + 0.3*(pixelHSV[2]-colorsHSV[color][2])**2)
            scores[color] += 1 / (np.power(2, dist/9))
    
    sorted_scores = dict(sorted(scores.items(), key=(lambda x: x[1]), reverse=True))

    total = 0
    for color in list(sorted_scores.keys())[:1]:
        total += sorted_scores[color]
    for color in list(sorted_scores.keys())[:1]:
        sorted_scores[color] /= total
        scores[color] = sorted_scores[color]
    
    for color in list(sorted_scores.keys())[1:]:
        scores[color] = 0
    
    return scores


class ShapeDescriptor:
    def __init__(self, image, segmentationMask):
        self.image = image 
        self.segmentationMask = segmentationMask
        self.masked_image = cv2.bitwise_and(self.image, self.image, mask=self.segmentationMask)
        self.cnt = cv2.findContours(self.segmentationMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #self.show(self.masked_image)]
        self.colorMask = np.zeros((self.image.shape[0], self.image.shape[1], 3), dtype=np.uint8)
        self.colorMask = cv2.cvtColor(self.colorMask, cv2.COLOR_BGR2HSV)
        self.colors = self.calculateAverageColors()
        
        self.primaryColor = colorsHSV[list(self.colors.keys())[0]]
        self.secondaryColor = colorsHSV[list(self.colors.keys())[1]]
        self.tertiaryColor = colorsHSV[list(self.colors.keys())[2]]
        
        self.compactness2D = self.calculateCompactness2D()
        self.roundness = self.calculateRoundness()
        self.HWRatio = self.calculateHWRatio()
        self.size = self.calculateSize()
        self.features = ["compactness2D", "roundness", "HWRatio", "size", "primaryColorH", "primaryColorS", "primaryColorV", "secondaryColorH", "secondaryColorS", "secondaryColorV"]
        self.score = 0
        self.descriptors = {}
        self.center = self.calculateCenter()


    def calculateCompactness2D(self):
        #Calculates ratio of area of segmentation mask to area of minimum bounding box
        boundingRect = cv2.minAreaRect(self.cnt[0][0])
        box = cv2.boxPoints(boundingRect)
        box = np.intp(box)
        side1 = math.dist(box[0], box[1])
        side2 = math.dist(box[1], box[2])
        boundingRectArea = side1 * side2
        area = cv2.countNonZero(self.segmentationMask)
        return area / boundingRectArea


    def calculateAverageColors(self):
        # Calculate the color of the object
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        masked_hsv = cv2.bitwise_and(hsv_image, hsv_image, mask=self.segmentationMask)
        pixels = masked_hsv[self.segmentationMask != 0]
        
        color_scores = {}
        for color in colorsHSV.keys():
            color_scores[color] = 0
        for pixel in pixels:
            scores = pixelColorScores(pixel)
            for color in colorsHSV.keys():
                color_scores[color] += scores[color]
        
        best_colors = dict(sorted(color_scores.items(), key=(lambda x: x[1]), reverse=True))
        #Normalize scores to sum to 1
        total_score = 0
        for color in best_colors.keys():
            total_score += best_colors[color]
        for color in best_colors.keys():
            best_colors[color] /= total_score
        
        minX = np.min(np.where(self.segmentationMask != 0)[1])
        maxX = np.max(np.where(self.segmentationMask != 0)[1])
        width = maxX - minX
        
        currX = minX
        for i, color in enumerate(best_colors.keys()):
            cv2.rectangle(self.colorMask, (int(currX), 0), (int(currX + width*best_colors[color]), self.colorMask.shape[0]), colorsHSV[color], -1)
            currX += width*best_colors[color]
        self.colorMask[self.segmentationMask == 0] = (0,0,20)

        return best_colors
        
    
    def calculateRoundness(self):
        #Calculates ratio of area of segmentation mask to area of minimum enclosing circle
        countour_perimeter = cv2.arcLength(self.cnt[0][0], True)
        (x,y), radius = cv2.minEnclosingCircle(self.cnt[0][0])
        area = cv2.countNonZero(self.segmentationMask)
        perimeter = np.pi * radius * 2
        actual_perimeterArea_ratio = countour_perimeter / area
        perimeterArea_ratio = perimeter / (np.pi * radius**2)
        
        return perimeterArea_ratio / actual_perimeterArea_ratio
    
    def calculateHWRatio(self):
        minRect = cv2.minAreaRect(self.cnt[0][0])
        box = cv2.boxPoints(minRect)
        box = np.intp(box)
        side1 = math.dist(box[0], box[1])
        side2 = math.dist(box[1], box[2])
        tempIm = np.zeros((self.image.shape[0], self.image.shape[1], 3), dtype=np.uint8)
        box = cv2.boxPoints(minRect)
        box = np.intp(box)
        return min(side1, side2) / max(side1, side2)

    def calculateSize(self):
        #Calculates the size of the object
        return cv2.countNonZero(self.segmentationMask)
        
    def updateDescriptors(self):
        self.descriptors["compactness2D"] = self.compactness2D
        self.descriptors["roundness"] = self.roundness
        self.descriptors["HWRatio"] = self.HWRatio
        self.descriptors["size"] = self.size

    def calculateCenter(self):
        #Calculates the center of the object
        centerX = np.mean(np.where(self.segmentationMask != 0)[1])
        centerY = np.mean(np.where(self.segmentationMask != 0)[0])
        return (centerX, centerY)

    def show(self, showMask=False, showColors=False, printDescriptors=False):
        if printDescriptors:
            self.updateDescriptors()
            print(self.descriptors)
        #self.masked_image[self.masked_image == 0] = 255
        cv2.imshow("object", self.masked_image)
        cv2.waitKey(0)
        cv2.destroyWindow("object")
        if showMask:
            if showColors:
                cv2.imshow("mask", cv2.cvtColor(self.colorMask, cv2.COLOR_HSV2BGR))

            else:           
                cv2.imshow("mask", cv2.cvtColor(self.colorMask, cv2.COLOR_HSV2BGR))
            cv2.waitKey(0)



class objectManager:
    def __init__(self, edgeThresh=50, minArea=30000):
        self.objects = []
        self.image = None
        self.masks = None
        self.edgeThresh = edgeThresh
        self.minArea = minArea
        self.megaMask = None
        self.weightsDict = {}

    def newImage(self, image, masks):
        self.image = image
        self.objects = []
        self.megaMask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        self.masks = masks
        self.cleanMasks()
        for mask in self.masks:
            this_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
            this_mask[mask['segmentation']] = (255,255,255)
            this_mask = cv2.cvtColor(this_mask, cv2.COLOR_BGR2GRAY)
            sd = ShapeDescriptor(image, this_mask)
            sd.show(showMask=True, showColors=True, printDescriptors=True)
            self.objects.append(sd)
    
    def cleanMasks(self):
        #remove any items that are too large
        self.masks = sorted(self.masks, key=(lambda x: x['area']), reverse=True)
        while self.masks[0]['area'] > self.minArea:
            self.masks.pop(0)

        #remove any items that are touching the edge of the image
        items_to_keep = []
        for i, mask in enumerate(self.masks):
            this_mask = np.zeros((self.image.shape[0], self.image.shape[1], 3), dtype=np.uint8)
            this_mask[mask['segmentation']] = (255,255,255)
            this_mask = cv2.cvtColor(this_mask, cv2.COLOR_BGR2GRAY)
            if np.any(this_mask[:self.edgeThresh] != 0) or np.any(this_mask[-self.edgeThresh:] != 0) or np.any(this_mask[:,:self.edgeThresh] != 0) or np.any(this_mask[:,-self.edgeThresh:] != 0):
                continue
            else:
                items_to_keep.append(i)
        new_masks = []
        for i in items_to_keep:
            new_masks.append(self.masks[i])

        #eliminate masks that are on top of each other
        items_to_keep = []
        existing_mask = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
        for i, mask in enumerate(new_masks):
            this_mask = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
            this_mask[mask['segmentation']] = 255
            if np.any(cv2.bitwise_and(existing_mask, this_mask) != 0):
                continue
            else:
                items_to_keep.append(i)
                existing_mask = cv2.bitwise_or(existing_mask, this_mask)
        
        final_masks = []
        for i in items_to_keep:
            final_masks.append(new_masks[i])
        
        self.masks = final_masks

    def generateMegaMask(self, show=False):
        for object in self.objects:
            self.megaMask[object.segmentationMask != 0] = (255,255,255)
        if show:
            cv2.imshow("mega_mask", self.megaMask)
            cv2.waitKey(20000)
            cv2.destroyWindow("mega_mask")

    def setWeightsDict(self, weightsDict):
        #check that every key in weightsDict is a valid feature
        for key in weightsDict.keys():
            assert key in (list(self.objects[0].features) + list(colorsHSV.keys()))
        self.weightsDict = weightsDict
    

    def __normalizeObjectSizes(self):
        #normalize the sizes of the objects so that the largest object has size 1
        max_size = 0
        for object in self.objects:
            if object.size > max_size:
                max_size = object.size
        for object in self.objects:
            object.size /= float(max_size)
            object.updateDescriptors()

    def __calculateScores(self):
        for object in self.objects:
            object_score = 0
            for key in self.weightsDict.keys():
                if key in colorsHSV.keys():
                    object_score += self.weightsDict[key] * object.colors[key]
                else:
                    object_score += self.weightsDict[key] * object.descriptors[key]
            object.score = object_score

    def __sortObjects(self):
        self.objects = sorted(self.objects, key=(lambda x: x.score), reverse=True)

    def getBestObject(self):
        self.__normalizeObjectSizes()
        self.__calculateScores()
        self.__sortObjects()
        return self.objects[0], self.objects[1], self.objects[2]
        




om = objectManager()
om.newImage(image, rgb_masks)
om.generateMegaMask(show=True)


print("Strawberries")
Strawberry_weightsDict = {
    "compactness2D": 0.3,
    "roundness": -.4,
    "HWRatio": -0.2,
    "size": 0.2,
    "DarkRed": 0.8
    #"Green": 0.2,
}

om.setWeightsDict(Strawberry_weightsDict)
best, nextBest, thirdBest = om.getBestObject()
print(best.center)
best.show()
nextBest.show()
thirdBest.show()

print("Blackberries")
# Blackberry_weightsDict = {
#     "compactness2D": -.2,
#     "roundness": -.2,
#     "HWRatio": 1.0,
#     "size": -.5,
#     "Black": 1.0,
# }
Blackberry_weightsDict = {
    "compactness2D": 0.5,    # Reflecting their somewhat compact but irregular shape
    "roundness": 0.4,        # Generally round, but not perfectly due to the small drupelets
    "HWRatio": 0.2,          # Slightly elongated due to the aggregated drupelets
    "size": -0.8,            # Small in size
    "DarkPurple": 1.0,       # Predominantly dark purple when ripe
    "Black": 0.5             # Sometimes appear almost black
}

om.setWeightsDict(Blackberry_weightsDict)
best, nextBest, thirdBest = om.getBestObject()
print(best.center)
print(nextBest.center)
print(thirdBest.center) 
best.show()
nextBest.show()
thirdBest.show()

print("Raspberries")
Raspberry_weightsDict = {
    "compactness2D": 0.6,
    "roundness": 0.2,
    "HWRatio": 0.3,
    "size": -1.0,
    "Red": 0.8,
    "LightRed": 0.3
}

om.setWeightsDict(Raspberry_weightsDict)
best, nextBest, thirdBest = om.getBestObject()
print(best.center)
best.show()
nextBest.show()

print("Grapes")
Grape_weightsDict = {
    "compactness2D": 0.0,
    "roundness": 0.0,
    "HWRatio": 0,
    "size": 0,
    "LightGreen": 0.8,
    "Gray": 0.4
}

om.setWeightsDict(Grape_weightsDict)
best, nextBest, thirdBest = om.getBestObject()
best.show()
print(best.center)
print(nextBest.center)
nextBest.show()
thirdBest.show()

print("Oranges")
Orange_weightsDict = {
    "compactness2D": 0.9,
    "roundness": 1.0,
    "HWRatio": 1.0,
    "size": 1.0,
    "Black": -1.0
}

om.setWeightsDict(Orange_weightsDict)
best, nextBest, thirdBest = om.getBestObject()
print(best.descriptors)
best.show()
nextBest.show()
print(best.center)

print("Carrots")
Carrot_weightsDict = {
    "compactness2D": -1.0,
    "roundness": -1.0,
    "HWRatio": -1.0,
    "size": .3,
    "Orange": 1.0,
    "LightOrange": 0.5
}

om.setWeightsDict(Carrot_weightsDict)
best, nextBest, thirdBest = om.getBestObject()
print(best.center)
print(nextBest.center)
best.show()
nextBest.show()




#Strawberry
#Banana
#Cellery
#Clementine
#Baby Carrot
#Tray

pitchLength = 0.07
pitchAngle = 0.0
yawAngle = 0.0
cameraHeight = 0.5
cameraXOffset = 0.01

camera_base_transform = np.eye(4)
translateZ1 = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, -pitchLength],
                       [0, 0, 0, 1]])
camera_base_transform = np.matmul(camera_base_transform, translateZ1)

rotateX = np.array([[1, 0, 0, 0],
                    [0, np.cos(pitchAngle), -np.sin(pitchAngle), 0],
                    [0, np.sin(pitchAngle), np.cos(pitchAngle), 0],
                    [0, 0, 0, 1]])
camera_base_transform = np.matmul(camera_base_transform, rotateX)

rotateY = np.array([[np.cos(yawAngle), 0, np.sin(yawAngle), 0],
                    [0, 1, 0, 0],
                    [-np.sin(yawAngle), 0, np.cos(yawAngle), 0],
                    [0, 0, 0, 1]])
camera_base_transform = np.matmul(camera_base_transform, rotateY)

new_Axis = np.array([[0, 0, 1, 0],
                    [-1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, 0, 1]])
camera_base_transform = np.matmul(camera_base_transform, new_Axis)

translateZ2 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, -cameraHeight],
                        [0, 0, 0, 1]])
camera_base_transform = np.matmul(camera_base_transform, translateZ2)

translateX = np.array([[1, 0, 0, -cameraXOffset],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
camera_base_transform = np.matmul(camera_base_transform, translateX)
