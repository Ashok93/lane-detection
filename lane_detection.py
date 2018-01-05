import numpy as np
import cv2

def grayscale(img):
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def show_image(img, img_name = "image"):
	cv2.imshow(img_name, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def blur(img, kernel_size=5):
	return cv2.GaussianBlur(img,(kernel_size, kernel_size),0)

def edge_detector(img, low_threshold, high_threshold):
	return cv2.Canny(img, low_threshold, high_threshold)

def detect_hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    hough_lines = cv2.HoughLinesP(img, rho, theta, threshold, minLineLength=min_line_len, maxLineGap=max_line_gap)
    return hough_lines

def average_slope_intercepts(lines):
	left_lines = [] # (slope, intercept)
	left_weights = [] # (length of line)
	right_lines = [] # (slope, intercept)
	right_weights = [] # (length of line)

	for line in lines:
		for x1,y1,x2,y2 in line:
			if x2 == x1 or y2 == y1: # ignoring vertical line
				continue

			slope = (y2-y1) / float((x2-x1))
			intercept = y1 - slope*x1
			length_of_line = np.sqrt((y2-y1)**2 + (x2-x1)**2)

			if slope < 0:
				left_lines.append((slope, intercept))
				left_weights.append(length_of_line)
			else:
				right_lines.append((slope, intercept))
				right_weights.append(length_of_line)


	left_lane  = np.dot(left_weights,  left_lines) /np.sum(left_weights)  if len(left_weights) >0 else None
	right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights)>0 else None
	
	return left_lane, right_lane

def convert_line_SI_points(y1, y2, line):

	if line is None:
		return None

	slope, intercept = line

	x1 = int((y1 - intercept)/slope)
	x2 = int((y2 - intercept)/slope)
	y1 = int(y1)
	y2 = int(y2)
	return (x1, y1, x2, y2)
    
def lane_lines(img, lines):
	left_lines, right_lines = average_slope_intercepts(lines)

	y1 = img.shape[0] # bottom of the image
	y2 = y1*0.6# slightly lower than the middle

	left_lines = convert_line_SI_points(y1, y2, left_lines)
	right_lines = convert_line_SI_points(y1, y2, right_lines)

	return left_lines, right_lines


if __name__ == "__main__":
	
	cap = cv2.VideoCapture('test_videos/solidWhiteRight.mp4')

	while(cap.isOpened()):

		ret, img = cap.read()
		gray_image = grayscale(img)
		blur_image = blur(gray_image, 5)
		edges = edge_detector(blur_image, 50, 150)

		mask = np.zeros_like(edges)
		ignore_mask_color = 255
		vertices = np.array([[(0,edges.shape[0]),(480, 310), (485, 310), (edges.shape[1],edges.shape[0])]], dtype=np.int32)
		cv2.fillPoly(mask, vertices, ignore_mask_color)
		masked_edges = cv2.bitwise_and(edges, mask)


		rho = 2 # distance resolution in pixels of the Hough grid
		theta = np.pi/180 # angular resolution in radians of the Hough grid
		threshold = 15    # minimum number of votes (intersections in Hough grid cell)
		min_line_length = 20 #minimum number of pixels making up a line
		max_line_gap = 15   # maximum gap in pixels between connectable line segments

		# Run Hough on edge detected image
		# Output "lines" is an array containing endpoints of detected line segments
		hough_lines = detect_hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

		left_line, right_line = lane_lines(blur_image, hough_lines)

		#left lane
		if left_line is not None:
			x1,y1,x2,y2 = left_line
			cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

		#right lane
		if right_line is not None:
			x1,y1,x2,y2 = right_line
			cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

		cv2.imshow("Video", img)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()