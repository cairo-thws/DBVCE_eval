# (c) Alexander Frühwald 2023
# This script generates the images for the paper.
# It takes the images from the imagenet_sample.py script and combines them into a single image.
# You have to download the result images (with 2x50 images) manually from wandb and put them in this folder.
# Name them as follows: {category}_{classifier}.png



from PIL import Image, ImageDraw, ImageFont

# Images must be in the same folder as this script
# The images must be named as follows:
# {category}_{classifier}.png

classifiers = [  "madry", "alex", "unet",  "conv", "swin", "simclr", "random"]

target_row = 5
left_margin = 0

categories = ["alp", "tiger"]

font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=25)

classifier_names = dict(
    madry="Madry",
    alex="AlexNet",
    unet="U-Net",
    conv="ConvNeXt",
    swin="Swin Transformer",
    simclr="SimCLR",
    random="RandomNet",
)
padding_x = 10

for category in categories:
    filenames = [f"{category}_{classifier}.png" for classifier in classifiers]
    # load all images into a list
    images = [Image.open(f"{filename}") for filename in filenames]
    # each of these images contains two columns and 50 rows
    # we only want the image of row `target_row` in the 2nd column
    
        
    # add the original to the beginning of the list.
    # the original is left image of the target_row
    original_img = images[0].crop((0, target_row * images[0].height // 50, images[0].width // 2, (target_row + 1) * images[0].height // 50))
    
    images = [original_img] + [image.crop((image.width // 2, target_row * image.height // 50, image.width, (target_row + 1) * image.height // 50)) for image in images]

    
    # now concatenate them horizontally
    result = Image.new("RGBA", (images[0].width * len(images) + left_margin+padding_x *( len(images) -1 ), images[0].height+30), (0,0,0,0))
    
    for index, image in enumerate(images):
        result.paste(im=image, box=(left_margin + index * images[0].width + index * padding_x, 0))
        
    # add the classifier_names to the image, below the image
    draw = ImageDraw.Draw(result)
    
    cls_names = ["Original"] + [classifier_names[classifier] for classifier in classifiers]
    
    for index, classifier in enumerate(cls_names):
        draw.text((left_margin + index * images[0].width + index * padding_x, images[0].height),
                  classifier, fill=(255, 255, 255), font=font)
           
        
    # save
    result.save(f"result_{category}.png")

    
        