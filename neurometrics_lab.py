import cv2
import numpy as np
import base64
import random

from sklearn.cluster import KMeans
from imutils.object_detection import non_max_suppression

import tensorflow as tf
from tensorflow.keras import backend as K
import gc


#############################################
## LOAD GRAPH MODELS
#############################################
#sess = tf.Session()
graph = tf.get_default_graph()
#K.set_session(sess)

#model_name = "model_%s_%s.pb" % (dataset, device)
model_name = "model_salicon_cpu.pb"
graph_def = tf.GraphDef()
with tf.gfile.Open(model_name, "rb") as file:    
    graph_def.ParseFromString(file.read())

## define the placeholder for image
input_img = tf.placeholder(tf.float32, (None, None, None, 3))

## define the input and output of the graph
[predicted_maps] = tf.import_graph_def(graph_def,
                                input_map={"input": input_img},
                                return_elements=["output:0"])

###########################################
##### SALIENCY MAP PREDICTION
###########################################
def saliency_model(img):       
    ## read any image
    h, w = img.shape[0], img.shape[1]
    img_input = cv2.resize(img, (320, 240))
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
    img_input = img_input[np.newaxis, :, :, :] ## reshape image (1, 240, 320, 3)

    ## do the prediction with the saliency model
    tf.keras.backend.clear_session()
    tf.contrib.keras.backend.clear_session()
    
    global sess
    global graph
    with graph.as_default():
        with tf.Session() as sess:
            #K.set_session(sess)
            ## send image to the graph
            saliency = sess.run(predicted_maps,
                                feed_dict={input_img: img_input})

            ## reshape image (240, 320, 3)
            saliency = cv2.cvtColor(saliency.squeeze(),
                                    cv2.COLOR_GRAY2BGR)
            saliency = np.uint8(saliency * 255)
            
            
            gc.collect()
        
        gc.collect()
        
    saliency = cv2.resize(saliency, (w, h))

    tf.keras.backend.clear_session()
    tf.contrib.keras.backend.clear_session()
    gc.collect()
    
    return saliency


###########################################
##### TEXT DETECTOR IN IMAGES
###########################################

def east_detector(img, umbral, net):   
    name = img
    scale = 448
    ##### PREPROCESS THE IMAGE #####       
    blob = cv2.dnn.blobFromImage(image = img,
                                 scalefactor = 1.0,
                                 size = (scale, scale),
                                 mean = (123.68, 116.78, 103.94),
                                 swapRB = True,
                                 crop = False)
    
    ##### CONFIG THE EAST MODEL #####
    ## get the loutput layers
    output_layers = []
    output_layers.append("feature_fusion/Conv_7/Sigmoid")
    output_layers.append("feature_fusion/concat_3")
    
    ##### PASS THE BLOB THROUGH THE NET    
    ## pass the blob through the network
    net.setInput(blob)
    
    ## get the scores and the geometry
    (scores, geometry)  = net.forward(output_layers)    
    #scores = output[0]
    #geometry = output[1]
    
    ##### GET THE BOXES AND PROBABILITIES #####
    num_rows, num_columns = scores.shape[2:4]
    confidences = []
    boxes = []
    ## navigate through every row
    for y in range(0, num_rows):
        ## get the probabilities
        scores_data = scores[0,0,y]

        ## get the bounding box coordinates
        x_0 = geometry[0, 0, y]
        x_1 = geometry[0, 1, y]
        x_2 = geometry[0, 2, y]
        x_3 = geometry[0, 3, y]

        ## get the angles
        angles_data = geometry[0, 4, y]

        ## navigate through every column
        for x in range(0, num_columns):
            ## if the score does not have sufficient probability, ignore
            if scores_data[x] < umbral:
                continue

            ## EAST naturally reduces the images in 4x
            (offset_x, offset_y) = (x * 4, y * 4)

            ## extract the rotation
            angle = angles_data[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            ## get the height and width
            h = x_0[x] + x_2[x]
            w = x_1[x] + x_3[x]

            ## starting and ending (x, y)           
            end_x = int(offset_x + (cos * x_1[x]) + (sin * x_2[x]))
            end_y = int(offset_y - (sin * x_1[x]) + (cos * x_2[x]))
            
            start_x = int(end_x - w)
            start_y = int(end_y - h)

            ## add the bounding box coordinates
            boxes.append((start_x, start_y, end_x, end_y))

            ## add the probabilities score
            confidences.append(scores_data[x])      
    
    ##### SUPRESSION NO MAXIMA
    boxes_nms = non_max_suppression(np.array(boxes), probs=confidences)
    
    ## do a copy to the original image
    img_copy = img.copy()
    
    ## get text area
    text_area = 0
        
    try:        
        ## navigate through the bounding boxes
        for (start_x, start_y, end_x, end_y) in boxes_nms:

            start_x = int(start_x * img.shape[1]/scale)
            start_y = int(start_y * img.shape[0]/scale)
            end_x = int(end_x * img.shape[1]/scale)
            end_y = int(end_y * img.shape[0]/scale)

            ## draw the bounding box on the image
            cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 10)

            ## crop the rectangle of the text
            img_text = img_copy[start_y:end_y, start_x:end_x]
            
            ## get text area
            text_area += img_text.shape[0] * img_text.shape[1]
                              
    except:
        raise

    ## get text area over image
    text_area_over_image = round(text_area / (img_copy.shape[0] * img_copy.shape[1]) * 100, 2)   

    
    return img, text_area_over_image, boxes_nms

###########################################
##### BLURRY DETECTOR IN IMAGES
###########################################

def blurry_detector(img, treshold=150):
    ## convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ## get variance of Laplace
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    text = "Not Blurry"
    perc_enhance = 0
    
    # if the focus measure is less than the supplied threshold,
    # then the image should be considered "blurry"
    if fm < treshold:
        text = "Blurry"  
        perc_enhance = 100 - round((fm / treshold) * 100, 2)
    
    
    return fm, text, perc_enhance

###########################################
##### TEXT OVER SALIENCY ESTIMATOR
###########################################
def text_area_over_saliency(img, boxes, umbral_attention, umbral_text):  
    ## convert to grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ## get the proportion of white and black pixel
    white_pixel = np.sum(gray >= umbral_attention)
    black_pixel = np.sum(gray < umbral_attention)
    per_white_pix = round(white_pixel / (gray.shape[0] * gray.shape[1]) * 100, 2)
    per_black_pix = round(black_pixel / (gray.shape[0] * gray.shape[1]) * 100, 2)

    ## do a copy to the original image
    img_copy = gray.copy()
    ## variables for save values
    white_pix_over_text = 0
    black_pix_over_text = 0
    area_over_text = 0
    
    if len(boxes) > 0:        
        ## navigate throught every text box
        for (start_x, start_y, end_x, end_y) in boxes:
            start_x = int(start_x * img.shape[1]/448)
            start_y = int(start_y * img.shape[0]/448)
            end_x = int(end_x * img.shape[1]/448)
            end_y = int(end_y * img.shape[0]/448)

            ## draw the bounding box on the image
            #cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)

            ## get the portion box of the image
            img_text = img_copy[start_y:end_y, start_x:end_x]

            ## get the proportion of white and black pixel in each text box
            white_pix_over_text += np.sum(img_text >= umbral_text)
            black_pix_over_text += np.sum(img_text < umbral_text)
            #area_over_text += img_text.shape[0] * img_text.shape[1]

        ## get the proportion of white and black pixel of text boxes over image
        perc_white_pix_over_text = round(white_pix_over_text / white_pixel * 100, 2)
        perc_black_pix_over_text = round(black_pix_over_text / black_pixel * 100, 2)
        
    else:
        perc_white_pix_over_text = 0.0
        perc_black_pix_over_text = 0.0        
        
    return img, per_white_pix, per_black_pix, perc_white_pix_over_text, perc_black_pix_over_text

###########################################
##### FOCUS MAP
###########################################
def focus_map(img, saliency, umbral_attention):
    ## get h, w of the saliency
    h, w = saliency.shape[0], saliency.shape[1]

    ## reshape original image
    img = cv2.resize(img, (saliency.shape[1], saliency.shape[0]))

    ## build and white image
    blanck_img = np.zeros([h,w,3], dtype=np.uint8)
    blanck_img.fill(255) # or img[:] = 255

    ## get white pixels
    white_indices = np.where(saliency >= umbral_attention)

    ## get the coordinates x, y
    coordinates = zip(white_indices[0], white_indices[1])
    
    ## fill white image with attention original image
    for x, y in coordinates:
        blanck_img[x, y] = img[x, y]
        
    return blanck_img

###########################################
##### HEATMAP
###########################################
def image_to_heatmap(img, saliency):
    ## get h, w of the saliency
    h, w = saliency.shape[0], saliency.shape[1]

    ## reshape original image
    img = cv2.resize(img, (saliency.shape[1], saliency.shape[0]))

    ## build and white image
    blanck_img = np.zeros([h,w,3], dtype=np.uint8)
    blanck_img.fill(255) # or img[:] = 255

    ################################################################### 5
    ## get white pixels
    indices = np.where((saliency >= 20) & (saliency < 50))

    ## get the coordinates x, y
    coordinates = zip(indices[0], indices[1])

    ## fill white image with attention original image
    for x, y in coordinates:
         blanck_img[x, y] = [192, 255, 51]

    ################################################################### 4
    ## get white pixels
    indices = np.where((saliency >= 50) & (saliency < 100))

    ## get the coordinates x, y
    coordinates = zip(indices[0], indices[1])

    ## fill white image with attention original image
    for x, y in coordinates:
        blanck_img[x, y] = [51, 255, 175]

    ################################################################### 3
    ## get white pixels
    indices = np.where((saliency >= 100) & (saliency < 150))

    ## get the coordinates x, y
    coordinates = zip(indices[0], indices[1])

    ## fill white image with attention original image
    for x, y in coordinates:
        blanck_img[x, y] = [51, 216, 255]

    ################################################################### 2
    ## get white pixels
    indices = np.where((saliency >= 150) & (saliency < 200))

    ## get the coordinates x, y
    coordinates = zip(indices[0], indices[1])

    ## fill white image with attention original image
    for x, y in coordinates:
        blanck_img[x, y] = [51, 87, 255]

    ################################################################### 1
    ## get white pixels
    indices = np.where(saliency >= 200)

    ## get the coordinates x, y
    coordinates = zip(indices[0], indices[1])

    ## fill white image with attention original image
    for x, y in coordinates:
        blanck_img[x, y] = [20, 20, 160]

    # apply the overlay
    heatmap_overlay = cv2.addWeighted(blanck_img, 0.5, img, 0.5, 0, img)
    
    return heatmap_overlay


###########################################
##### GET DOMINANT COLORS
###########################################
def dominant_colors(img):
    ## pre-process the image
    img = cv2.resize(img, (100, 200))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((img.shape[0] * img.shape[1], 3))

    ## train the model
    clt = KMeans(5)
    clt.fit(img)
    
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # loop over the percentage of each cluster and the color of
    # each cluster
    colors = []
    percentages = []
    for (percent, color) in zip(hist, clt.cluster_centers_):
        colors.append([color.tolist(), percent])
        
    return colors


###########################################
##### IMAGE TO BASE64
###########################################
def image_to_base64(img):
    ## transform the image to base64
    retval, buffer = cv2.imencode('.jpg', img)
    b64_string = base64.b64encode(buffer)
    b64_string = b64_string.decode("utf-8")
    #b64_format = "data:image/jpg;base64," + b64_string
    b64_format = b64_string
    
    return b64_format



###########################################
##### GET AREA OF INTEREST
###########################################
def get_aoi(original_img, overlay_img, coordinates, index_aoi, umbral_attention=1):
    ## read image from URL
    #img = io.imread(img_url)   
    h = overlay_img.shape[0] ## height of the image
    w = overlay_img.shape[1] ## width of the image
    
    ## read coordinates
    points = np.array(coordinates)
    
    ## random color
    color_list = [(55, 187, 255), (246, 218, 98), (194, 255, 0), (204, 0, 255), (0, 255, 255), (219, 152, 52),
                  (113, 204, 46), (182, 89, 155), (15, 196, 241), (176, 201, 72), 
                  (76, 0, 255), (0, 204, 255), (255, 189, 0), (0, 95, 255), (60, 76, 231),
                  (223, 255, 0), (0, 255, 239), (159, 231, 149), (241, 214, 174), (222, 180, 210),
                  (117, 141, 19), (91, 24, 194), (59, 235, 255), (177, 143, 244), (0, 111, 255),
                  (247, 195, 79), (255, 255, 0), (153, 0, 255), (248, 237, 145), (192, 231, 97)]
    
    ## navigate by each coordinates
    list_per_white_pix = []
       
    for p in points:
        img_copy = overlay_img.copy()
        ## create a mask ROI of the original image
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [p], (255))
        roi = cv2.bitwise_and(img_copy,img_copy,mask = mask)

        ## convert image RGB to GRAY
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        ## get the proportion of white and black pixel
        white_pixel = np.sum(roi_gray >= umbral_attention)       
        per_white_pix = round(white_pixel / (h * w) * 100, 2)
               
        ## add to the list 
        list_per_white_pix.append(per_white_pix)
        
        ## add polylines
        #color = random.choice(color_list)
        color = color_list[index_aoi]
        cv2.polylines(original_img, np.int32([p]), 1, color, 10)
        
    return list_per_white_pix, original_img

def sift_n_fast(imagen):
    return imagen