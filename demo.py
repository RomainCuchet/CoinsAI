from acs import Acs
from tools import Tools
image_path = "datasets/baht_own_dataset/images/c6440c0b-baht_d9.jpg" # change with to your own image path
cam_acs = Acs("models/yolov8s_coinai.pt")
image_path,nb_coins,nb_circles,value,reclassified = cam_acs.get_prediction(image_path,(0,0),(1199,1599))
Tools.show_results_tkinter(image_path,nb_coins,nb_circles,value,reclassified)