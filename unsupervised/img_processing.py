from openslide import OpenSlide

svs_path = 'Slide 10.svs'

# Open whole slide image
img = OpenSlide(svs_path)

# Get shape
original_dimensions = img.dimensions

# Get thumbnail image (t times smaller)
t = 512
resize_width = original_dimensions[0]/t
resize_height = original_dimensions[1]/t
resized_img = img.get_thumbnail((resize_width,resize_height)) # PIL.Image type
resized_img.save('thumbnail_images_sample/'+svs_path.split('.')[0]+'.png')
# Get a region of the original image (here [10000:10100,10000:10100])
cropped_img = img.read_region((10000,10000),0,(100,100)) # PIL Image RGBA
cropped_img = cropped_img.convert('RGB') # PIL Image RGB
cropped_img.save('tmp.png')
