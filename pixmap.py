from flask import Flask, render_template, request, jsonify, make_response
from flask_restful import Resource, Api
from flask_cors import CORS

import base64
import numpy as np
import cv2
import json
import gc
import tensorflow as tf

#import neurometrics_lab

## create an instace of the app
app = Flask(__name__)
app.config['CORS_HEADERS'] = 'content-Type'
CORS(app)

api = Api(app)

## laod model
##### LOAD THE EAST MODEL #####    
## now let's go loading the network from TensorFlow format
model = "frozen_east_text_detection.pb"
net = cv2.dnn.readNetFromTensorflow(model)

class Neurometrics(Resource):
    def get(self):
        return("Pixmap-Neurometrics")

    def post(self):
        tf.keras.backend.clear_session()
        tf.contrib.keras.backend.clear_session()
        gc.collect()
        
        ## GET IMAGE
        img = request.json["image"]
        print(type(img))
        h = request.json["h"]
        w = request.json["w"]  
        #header, encoded = str(img).split(",", 1)
        b64_str = base64.b64decode(img)
        nparr = np.frombuffer(b64_str, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        #img_np = cv2.resize(img_np, (400, 300))
             
        ## GET SALIENCY MAP IMAGE
        saliency_img = neurometrics_lab.saliency_model(img_np)       
        
        ## GET TEXT IN IMAGE
        img_for_east = img_np.copy()
        img_text_detection, text_area_over_image, boxes_nms = neurometrics_lab.east_detector(img=img_for_east, umbral=0.99, net=net)
        
        ## image to base64
        #img_text_detection = cv2.resize(img_text_detection, (w, h))
        img_text_detection = neurometrics_lab.image_to_base64(img_text_detection)
        
        ##########################################
        ## GET BLURRY
        ##########################################
        img_for_blurry = img_np.copy()
        fm, text, perc_enhance = neurometrics_lab.blurry_detector(img_for_blurry, 150)
        
        ##########################################
        ## GET PERC WHITE AND BLACK OVER TEXT
        ##########################################
        saliency_for_text = saliency_img.copy()
        img_, per_white_pix, per_black_pix, perc_white_pix_over_text, perc_black_pix_over_text = neurometrics_lab.text_area_over_saliency(saliency_for_text, boxes_nms, 20, 20)
        
        ##########################################
        ## GET FOCUS MAP
        ##########################################
        img_for_focus = img_np.copy()
        saliency_for_focus = saliency_img.copy()
        #print(saliency_for_focus.shape)
        focus_img = neurometrics_lab.focus_map(img_for_focus, saliency_for_focus, umbral_attention=20)
        
        ## image to base64
        #focus_img = cv2.resize(focus_img, (w, h))
        focus_img = neurometrics_lab.image_to_base64(focus_img)
        
        ##########################################
        ## HEATMAP
        ##########################################
        img_for_heatmap = img_np.copy()
        saliency_for_heatmap = saliency_img.copy()
        heatmap_overlay = neurometrics_lab.image_to_heatmap(img_for_heatmap, saliency_for_heatmap)
        
        ## image to base64
        #heatmap_overlay = cv2.resize(heatmap_overlay, (w, h))
        heatmap_overlay = neurometrics_lab.image_to_base64(heatmap_overlay)
        
        ##########################################
        ## OVERLAY
        ##########################################
        ## image to base64 
        img_for_saliency = img_np.copy()        
        alpha = saliency_img.astype(float)/255
        img_for_saliency = img_for_saliency.astype(float)/255
        overlay = cv2.multiply(alpha, img_for_saliency)

        #saliency_img = cv2.resize(overlay, (w, h)) * 255
        saliency_img = overlay * 255
        saliency_img = neurometrics_lab.image_to_base64(saliency_img)
        
        ##########################################
        ## DOMINANT COLORS
        ##########################################
        dominant_colors = neurometrics_lab.dominant_colors(img_np)
        
        ##########################################
        ## CTR PREDICTION
        ##########################################
        sift_fast = neurometric_lab.sift_n_fast(img_np)

        c1 = kp_sift[0]
        c2 = kp_sift[1]
        c3 = kp_sift[2]
        c4 = kp_sift[3]
        c5 = kp_sift[4]
        c6 = kp_sift[5]
        c7 = kp_sift[6]
        c8 = kp_sift[7]
        c9 = kp_sift[8]
        c10 =kp_sift[9]
        c11 =kp_sift[10]
        c12 =kp_sift[11]
        c13 =kp_sift[12]
        c14= kp_sift[13]
        c15 =kp_sift[14]
        c16= kp_sift[15]
        c17= kp_sift[16]
        c18= kp_sift[17]
        c19= kp_sift[18]
        c20= kp_sift[19]
        c21= kp_sift[20]
        c22= kp_sift[21]
        c23= kp_sift[22]
        c24= kp_sift[23]
        c25= kp_sift[24]


        c1_fast = kp_fast[0]
        c2_fast = kp_fast[1]
        c3_fast = kp_fast[2]
        c4_fast = kp_fast[3]
        c5_fast = kp_fast[4]
        c6_fast = kp_fast[5]
        c7_fast = kp_fast[6]
        c8_fast = kp_fast[7]
        c9_fast = kp_fast[8]
        c10_fast = kp_fast[9]
        c11_fast = kp_fast[10]
        c12_fast = kp_fast[11]
        c13_fast = kp_fast[12]
        c14_fast= kp_fast[13]
        c15_fast = kp_fast[14]
        c16_fast= kp_fast[15]
        c17_fast= kp_fast[16]
        c18_fast= kp_fast[17]
        c19_fast= kp_fast[18]
        c20_fast= kp_fast[19]
        c21_fast= kp_fast[20]
        c22_fast= kp_fast[21]
        c23_fast= kp_fast[22]
        c24_fast= kp_fast[23]
        c25_fast= kp_fast[24]

        tabla = []
        tabla.append([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,
        c1_fast,c2_fast,c3_fast, c4_fast, c5_fast, c6_fast, c7_fast, c8_fast, c9_fast, c10_fast,c11_fast,c12_fast, 
        c13_fast, c14_fast, c15_fast, c16_fast,c17_fast,c18_fast,c19_fast, c20_fast, c21_fast, c22_fast, c23_fast, c24_fast, c25_fast])
        
        df = pd.DataFrame(tabla, columns=["c1_sift", "c2_sift", "c3_sift", "c4_sift", "c5_sift", "c6_sift", "c7_sift", "c8_sift", "c9_sift",
                                    "c10_sift", "c11_sift", "c12_sift", "c13_sift", "c14_sift", "c15_sift", "c16_sift",
                                    "c17_sift", "c18_sift", "c19_sift", "c20_sift", "c21_sift", "c22_sift", "c23_sift", "c24_sift", "c25_sift",

                                    "c1","c2","c3","c4","c5","c6","c7","c8","c9","c10","c11","c12","c13","c14","c15","c16","c17","c18","c19","c20",
                                    "c21","c22","c23","c24","c25"])

        #Aca se coloca exactamente lo mismo que lo que probe
        #en local, es decir, se agrega el pickle
        #luego se normaliza y finalmente se usa
        #el prefict, eso se manda al API

        response = jsonify({"text_area_over_image": text_area_over_image,
                            "text_area_over_saliency": perc_white_pix_over_text,
                            "per_enhance": perc_enhance,
                            "per_attention": per_white_pix,
                            "per_non_attention": per_black_pix,
                            "image_saliency": saliency_img,
                            "image_text_detection": img_text_detection,
                            "image_focus_map": focus_img,
                            "image_heatmap": heatmap_overlay,
                            "dominant_colors": dominant_colors})
        
        del text_area_over_image
        del perc_white_pix_over_text
        del perc_enhance
        del per_white_pix
        del per_black_pix
        del saliency_img
        del img_text_detection
        del focus_img
        del heatmap_overlay
        del dominant_colors       
               
        tf.keras.backend.clear_session()
        tf.contrib.keras.backend.clear_session()
        gc.collect()
        #{"text_area_over_image":text_area_over_image}
        return make_response(response, 200)
    
############################################
## ENDPOINT FOR AOI
############################################
class AoI(Resource):    
    ## get function
    def get(self):
        return("Pixmap-AOi")
    
    ## put function
    def post(self):
        ## get coordinates
        coordinates = request.json["coordinates"]
        
        ## get original image
        original_img = request.json["original_img"] ## get image url 
        b64_str_original = base64.b64decode(original_img) ## get image in base64
        nparr_original = np.frombuffer(b64_str_original, np.uint8) ## transform image 
        original_img = cv2.imdecode(nparr_original, cv2.IMREAD_COLOR) ## transform image
        
        ## get overlay image
        overlay_img = request.json["overlay_img"] ## get image url        
        b64_str_overlay = base64.b64decode(overlay_img) ## get image in base64
        nparr_overlay = np.frombuffer(b64_str_overlay, np.uint8) ## transform image 
        overlay_img = cv2.imdecode(nparr_overlay, cv2.IMREAD_COLOR) ## transform image                  
        
        ## get index_aoi
        index_aoi = request.json["index_aoi"]
        
        ##########################################
        ## GET AOI
        ##########################################
        per_white_pix, roi_image = neurometrics_lab.get_aoi(original_img, overlay_img, coordinates, index_aoi, umbral_attention=1)
        
        ## image to base64      
        roi_image = neurometrics_lab.image_to_base64(roi_image)
        
        ## make response in JSON format
        response = jsonify({"per_attention": per_white_pix,
                            "roi_image": roi_image})      
        
        return make_response(response, 200)
    
#############################################
## DEFINE THE ENDPOINTS
#############################################
api.add_resource(Neurometrics, "/neurometrics")
api.add_resource(AoI, "/aoi")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
