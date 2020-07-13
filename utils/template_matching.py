import cv2
import matplotlib.image as mpimg

def find_boxes(img, templates):
    bbox_list = []
    method = eval('cv2.TM_CCOEFF_NORMED')
    for template_img_path in templates:
        template = mpimg.imread(template_img_path)
        
        # Apply template Matching
        result = cv2.matchTemplate(img, template, method)
        # Use cv2.minMaxLoc() to extract the location of the best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        h, w, _ = template.shape
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        bbox_list.append((top_left, bottom_right))
       
    return bbox_list