import numpy as np
import cv2
import random

def get_samples(images, bboxes, cl, size, class_map, get_negatives=True):
    '''
    Returns bounding box cropped images (+ negative 
    background patches depending on job), class labels, 
    and the bounding box coordinates
    '''
    # Constants used for overlap range
    e = 50
    resize_buffer = 20
    rendered_img_size = images.shape[1]
    ov_range = 0.15*rendered_img_size
    
    xs = []
    ys = []
    bbs = []

    for row in range(images.shape[0]):
        img = images[row]
        x_min, x_max, y_min, y_max = bboxes[row]    
        pos_img = img[y_min:y_max, x_min:x_max]
        pos_img = cv2.resize(pos_img, (size, size))
        
        xs.append(pos_img)
        if class_map is None:
            ys.append(-2)
        else:
            ys.append(class_map[cl])
        bbs.append(bboxes[row])
        
        #if job is train then add negative samples of the data as well
        if get_negatives: 
            while (True): 
                # Get the negative bounding box from the image outside the 
                # original bounding box but with some overlap range
                # Randomly select 4 points within a window specified 
                # by rendered_img_size and e
                amin = random.randint(0, rendered_img_size - e)
                bmin = random.randint(0, rendered_img_size - e)
                # Adding the resize buffer to prevent resizing 
                # errors on very thin strips
                amax = random.randint(amin+resize_buffer, rendered_img_size) 
                bmax = random.randint(bmin+resize_buffer, rendered_img_size)

                a_ratio = float(amax-amin)/float(bmax-bmin)

                if 0.5 < a_ratio and a_ratio < 2:
                    if (overlap(x_min, y_min, x_max, y_max, 
                                amin, bmin, amax, bmax, ov_range)):
                        break

            neg_img = img[amin:amax, bmin:bmax] #create negative image
            
            neg_img = cv2.resize(neg_img, (size,size))
            
            xs.append(neg_img)
            ys.append(-1) # append -1 for negative sample
            bbs.append([bmin, bmax, amin, amax])

    x = np.array(xs, dtype=np.float32)
    y = np.array(ys)
    bb = np.array(bbs)
    
    x = x.reshape((x.shape[0], size, size, 3))

    # Make it 3xsizexsize 
    x = x.transpose(0,3,1,2)

    return [x,y,bb]

def overlap(xmin, ymin, xmax, ymax, amin, bmin, amax, bmax, ov_range):
    """
    Checks for the rectangular collision for 
    the two boxes with a certain overlap range.
    Returns True if the two boxes are valid 
    and do not overlap more than the allowed range
    """
    if amax < (xmin+ov_range): 
        if bmax < (ymin+ov_range) or bmin > (ymax-ov_range):
            return True
    if amin > xmax - ov_range:
        if bmax < (ymin+ov_range) or bmin > (ymax-ov_range):
            return True
    return False