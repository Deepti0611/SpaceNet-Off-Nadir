The SpaceNet satellite imagery is stored in Amazon's S3 storage, and we used AWS CLI to download the actual TIFF image files to our computer.
1. Satellite imagery 900x900 pixels , 4 bands- red-blue-green-NIR(near infrared)
 SEMANTIC SEGMENTATION(what pixels contain the building)
 2. Buildin annotations- so first we had a csv file with PolygonWKT_Pix - contains building oolugons in pixel coordinates 
 this was used to turn those polygons into a mask 

ImageId
BuildingId
PolygonWKT_Pix- denotes building shape 
PolygonWKT_Geo

most imp one was-ImageId
PolygonWKT_Pix- because we needed to associate an image with its building locations and create a pixel mask.
3. All six images represent the same geographic tile from different viewing angles.
4.so we chose 20 from this-
5. from here split train,test,validation(70,15,15)
TRAIN
14 tiles × 6 = 84 images

VALIDATION
3 tiles × 6 = 18 images

TEST
3 tiles × 6 = 18 images
6. created building mask from the annotation -This tells us which pixels belong to a building.
rasterized the polygon- 0-background
255=building 
saved as png masks-the correct answer that we give the model during supervised 
learning.
During training, the model's output will eventually be compared against our generated ground-truth mask.
7. metadata.csv
info about
tile
angle
category
split
image_id
image_path
mask_path
building_count
image_width
image_height
num_bands
8. convert 4 bands to 3



