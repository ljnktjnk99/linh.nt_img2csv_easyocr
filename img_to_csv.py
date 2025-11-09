"""
Convert table image to CSV using EasyOCR with automatic skew correction
Uses Hough Line Transform for skew detection
"""
import easyocr
import time
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import argparse
import os


def detect_skew_angle(image_path, save_visualization=False):
	"""
	Detect skew angle using Hough Line Transform
	
	Args:
		image_path: Path to input image
		save_visualization: If True, save image with detected lines
	
	Returns:
		float: Skew angle in degrees
	"""
	img = cv2.imread(image_path)
	if img is None:
		return 0.0
	
	# Preprocessing
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (5, 5), 0)
	edges = cv2.Canny(blur, 30, 150, apertureSize=3)
	
	# Hough Line Transform
	lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=100, maxLineGap=10)
	
	if lines is None:
		return 0.0
	
	# Calculate angles from detected lines
	angles = []
	valid_lines = []
	
	for line in lines:
		x1, y1, x2, y2 = line[0]
		
		# Filter short lines
		length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
		if length < 50:
			continue
		
		# Calculate angle
		angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
		
		# Normalize to [-45, 45]
		if angle < -45:
			angle += 90
		elif angle > 45:
			angle -= 90
		
		# Only keep near-horizontal lines (text lines)
		if abs(angle) < 45:
			angles.append(angle)
			if save_visualization:
				valid_lines.append((x1, y1, x2, y2))
	
	if not angles:
		return 0.0
	
	median_angle = np.median(angles)
	
	# Save visualization if requested
	if save_visualization:
		debug_img = img.copy()
		for x1, y1, x2, y2 in valid_lines:
			cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
		
		cv2.putText(debug_img, f"Angle: {median_angle:.2f} deg", 
				   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
		
		base_name = os.path.splitext(os.path.basename(image_path))[0]
		debug_path = f"{base_name}_debug.png"
		cv2.imwrite(debug_path, debug_img)
		print(f"Debug image saved: {debug_path}")
	
	return median_angle


def rotate_image(image_path, angle, output_path=None):
	"""
	Rotate image to correct skew
	
	Args:
		image_path: Path to input image
		angle: Rotation angle in degrees
		output_path: Path to save rotated image (None = auto-generate)
	
	Returns:
		str: Path to rotated image
	"""
	img = Image.open(image_path)
	
	if output_path is None:
		base_name = os.path.splitext(os.path.basename(image_path))[0]
		output_path = f"{base_name}_deskewed.png"
	
	rotated = img.rotate(angle, expand=True, fillcolor='white', resample=Image.BICUBIC)
	rotated.save(output_path)
	
	return output_path


def analyze_table_structure(results):
	"""
	Analyze table structure from OCR results
	
	Args:
		results: List of (bbox, text, confidence) from EasyOCR
	
	Returns:
		pd.DataFrame: Table data
	"""
	if not results:
		return pd.DataFrame()
	
	# Extract coordinates
	data_points = []
	for bbox, text, confidence in results:
		x_coords = [p[0] for p in bbox]
		y_coords = [p[1] for p in bbox]
		
		data_points.append({
			'text': text,
			'center_x': sum(x_coords) / 4,
			'center_y': sum(y_coords) / 4,
			'height': max(y_coords) - min(y_coords)
		})
	
	if not data_points:
		return pd.DataFrame()
	
	# Sort by Y coordinate (top to bottom)
	data_points.sort(key=lambda x: x['center_y'])
	
	# Group into rows
	rows = []
	current_row = [data_points[0]]
	y_threshold = data_points[0]['height'] * 0.3
	
	for point in data_points[1:]:
		avg_y = sum(p['center_y'] for p in current_row) / len(current_row)
		
		if abs(point['center_y'] - avg_y) < y_threshold:
			current_row.append(point)
		else:
			current_row.sort(key=lambda x: x['center_x'])
			rows.append(current_row)
			current_row = [point]
			y_threshold = point['height'] * 0.3
	
	if current_row:
		current_row.sort(key=lambda x: x['center_x'])
		rows.append(current_row)
	
	# Build DataFrame
	if not rows:
		return pd.DataFrame()
	
	max_cols = max(len(row) for row in rows)
	table_data = []
	
	for row in rows:
		row_data = [p['text'] for p in row]
		row_data.extend([''] * (max_cols - len(row_data)))
		table_data.append(row_data)
	
	return pd.DataFrame(table_data)


def convert_img_to_csv(image_path, output_csv='output.csv', save_debug=False):
    """
    Convert table image to CSV with automatic skew correction
    
    Args:
        image_path: Path to input image
        output_csv: Path to output CSV file
        save_debug: Save debug visualization with detected lines
    
    Returns:
        tuple: (results, total_time, df, deskewed_path)
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return None, 0, pd.DataFrame(), None
    
    print(f"Processing: {image_path}")
    
    total_start = time.time()
    deskewed_path = None
    
    # Step 1: Detect skew using Hough Line Transform (independent of OCR)
    skew_angle = detect_skew_angle(image_path, save_visualization=save_debug)
    
    # Step 2: Rotate image if needed
    if abs(skew_angle) > 0.5:
        print(f"Detected skew: {skew_angle:.2f}°, correcting...")
        deskewed_path = rotate_image(image_path, skew_angle)
        image_to_ocr = deskewed_path
    else:
        image_to_ocr = image_path
    
    # Step 3: OCR once on the (possibly corrected) image
    print("Running OCR...")
    reader = easyocr.Reader(['ja', 'en'], gpu=False)
    results = reader.readtext(image_to_ocr, detail=1, paragraph=False)
    
    print(f"OCR detected {len(results)} text boxes")
    
    # Analyze table structure
    df = analyze_table_structure(results)
    
    if not df.empty:
        print(f"Table detected: {len(df)} rows × {len(df.columns)} columns")
        
        # Save CSV
        df.to_csv(output_csv, index=False, header=False, encoding='utf-8-sig')
        print(f"CSV saved: {output_csv}")
    else:
        print("Warning: No table structure detected")
    
    total_time = time.time() - total_start
    print(f"Total time: {total_time:.2f}s")
    
    return results, total_time, df, deskewed_path


def main():
	parser = argparse.ArgumentParser(
		description='Convert table image to CSV using EasyOCR with auto skew correction',
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  python img_to_csv.py table.png
  python img_to_csv.py table.png --output result.csv
  python img_to_csv.py table.png --debug
		"""
	)
	
	parser.add_argument('input', help='Path to input table image')
	parser.add_argument('--output', '-o', default='output.csv',
					   help='Path to output CSV file (default: output.csv)')
	parser.add_argument('--debug', action='store_true',
					   help='Save debug visualization with detected lines')
	
	args = parser.parse_args()
	
	# Run conversion
	results, total_time, df, deskewed_path = convert_img_to_csv(
		args.input,
		args.output,
		save_debug=args.debug
	)
	
	# Print result
	print("\n" + "=" * 50)
	if df is not None and not df.empty:
		print("SUCCESS")
		if deskewed_path:
			print(f"Deskewed image: {deskewed_path}")
	else:
		print("FAILED")
	print("=" * 50)


if __name__ == '__main__':
	main()
