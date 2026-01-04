import cv2
import numpy as np
import time

class SimpleShoppingCartTracker:
    def __init__(self):
        """Initialize the tracker"""
        print("Initializing Shopping Cart Tracker...")
        
        # Open webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not self.cap.isOpened():
            raise Exception("Could not open webcam!")
        
        print("✅ Camera connected!")
        
        # Background reference
        self.reference_gray = None
        self.is_calibrated = False
        
        # Zones
        self.zone_divider = 0.6
        self.cart_zone_x = None
        
        # Cart tracking
        self.stable_objects = {}
        self.cart_count = 0
        self.item_price = 15.99
        
        # Detection settings
        self.min_area = 3000
        self.max_area = 40000
        self.diff_threshold = 30
        self.stability_frames = 15
        self.position_tolerance = 80
        
        # Skin filter
        self.enable_skin_filter = True
        
    def is_skin(self, roi):
        """Detect skin color to filter hands"""
        if not self.enable_skin_filter or roi.size == 0:
            return False
        try:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower = np.array([0, 20, 70], dtype=np.uint8)
            upper = np.array([20, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            return (np.count_nonzero(mask) / mask.size) > 0.4
        except:
            return False
    
    def calibrate(self, frame):
        """Take background snapshot"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.reference_gray = cv2.GaussianBlur(gray, (21, 21), 0)
        self.is_calibrated = True
        print("✅ CALIBRATED! Start using the tracker now!")
    
    def detect_objects(self, frame):
        """Detect objects"""
        if self.reference_gray is None:
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        delta = cv2.absdiff(self.reference_gray, gray)
        thresh = cv2.threshold(delta, self.diff_threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter odd shapes
            ratio = w / float(h) if h > 0 else 0
            if ratio > 3 or ratio < 0.3:
                continue
            
            # Filter skin
            roi = frame[y:y+h, x:x+w]
            if self.is_skin(roi):
                continue
            
            center = (x + w // 2, y + h // 2)
            objects.append({'center': center, 'bbox': (x, y, w, h)})
        
        return objects
    
    def update_cart(self, detected_objects, frame_width):
        """Update cart count"""
        if self.cart_zone_x is None:
            self.cart_zone_x = int(frame_width * self.zone_divider)
        
        # Find cart objects
        cart_objects = [obj for obj in detected_objects 
                       if obj['center'][0] > self.cart_zone_x]
        
        current_positions = {}
        for obj in cart_objects:
            key = (int(obj['center'][0] / self.position_tolerance), 
                   int(obj['center'][1] / self.position_tolerance))
            
            if key in self.stable_objects:
                self.stable_objects[key] += 1
            else:
                self.stable_objects[key] = 1
            
            current_positions[key] = obj
        
        # Remove old positions
        for key in list(self.stable_objects.keys()):
            if key not in current_positions:
                del self.stable_objects[key]
        
        # Count stable objects
        self.cart_count = sum(1 for v in self.stable_objects.values() 
                             if v >= self.stability_frames)
    
    def draw_ui(self, frame, detected_objects):
        """Draw all UI elements"""
        h, w = frame.shape[:2]
        
        if self.cart_zone_x is None:
            self.cart_zone_x = int(w * self.zone_divider)
        
        # Draw zone divider
        cv2.line(frame, (self.cart_zone_x, 0), (self.cart_zone_x, h), (255, 255, 0), 3)
        
        # Zone labels
        cv2.putText(frame, "TABLE", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, "CART", (self.cart_zone_x + 20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
        
        # Draw detected objects
        for obj in detected_objects:
            x, y, w_box, h_box = obj['bbox']
            center = obj['center']
            in_cart = center[0] > self.cart_zone_x
            
            key = (int(center[0] / self.position_tolerance), 
                   int(center[1] / self.position_tolerance))
            stability = self.stable_objects.get(key, 0)
            
            if in_cart:
                if stability >= self.stability_frames:
                    color = (0, 255, 255)
                    label = f"COUNTED ({stability}f)"
                    thickness = 3
                else:
                    color = (255, 128, 0)
                    label = f"WAIT {stability}/{self.stability_frames}"
                    thickness = 2
            else:
                color = (0, 255, 0)
                label = "TABLE"
                thickness = 2
            
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, thickness)
            cv2.circle(frame, center, 8, (0, 0, 255), -1)
            
            label_y = y - 10 if y > 30 else y + h_box + 25
            cv2.putText(frame, label, (x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Info panel
        overlay = frame.copy()
        panel_h = 200
        cv2.rectangle(overlay, (0, h - panel_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        total = self.cart_count * self.item_price
        
        y = h - panel_h + 40
        cv2.putText(frame, "SHOPPING CART", (30, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        y += 50
        cv2.putText(frame, f"Items: {self.cart_count}", (30, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        
        y += 50
        cv2.putText(frame, f"Total: ${total:.2f}", (30, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        # Help
        help_y = 90
        cv2.putText(frame, "Q/ESC - Quit", (w - 220, help_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "R - Reset", (w - 220, help_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "C - Recalibrate", (w - 220, help_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def run(self):
        """Main loop"""
        print("\n" + "="*70)
        print("SIMPLE SHOPPING CART TRACKER")
        print("="*70)
        print("\n🎯 QUICK START:")
        print("1. Window will open")
        print("2. Remove all objects from view")
        print("3. Press SPACE (or click the window) to calibrate")
        print("4. Place objects and move them to cart!")
        print("\n⌨️  CONTROLS:")
        print("  SPACE or CLICK - Calibrate background")
        print("  Q or ESC - Quit")
        print("  R - Reset cart")
        print("  C - Recalibrate")
        print("\n💡 TIP: Make sure to CLICK the window before pressing keys!")
        print("="*70 + "\n")
        
        window_name = 'Shopping Cart Tracker'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Mouse callback for calibration
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and not self.is_calibrated:
                print("🖱️  Mouse click detected - calibrating...")
        
        cv2.setMouseCallback(window_name, mouse_callback)
        
        print("⏳ Starting in 2 seconds...")
        time.sleep(2)
        
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Check for calibration
            if not self.is_calibrated:
                # Show calibration prompt
                flash = (frame_count // 15) % 2 == 0
                color = (0, 255, 255) if flash else (100, 200, 200)
                
                cv2.putText(frame, "READY TO CALIBRATE", (w//2 - 200, h//2 - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                cv2.putText(frame, "Remove all objects", (w//2 - 150, h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, "Then press SPACE or CLICK", (w//2 - 200, h//2 + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                cv2.imshow(window_name, frame)
                
                key = cv2.waitKey(30)
                
                # Calibrate on SPACE or ENTER
                if key == 32 or key == 13:  # SPACE or ENTER
                    print("\n📸 Taking background snapshot...")
                    time.sleep(0.5)  # Brief pause
                    # Capture fresh frame
                    ret, fresh_frame = self.cap.read()
                    if ret:
                        fresh_frame = cv2.flip(fresh_frame, 1)
                        self.calibrate(fresh_frame)
                
                # Quit
                elif key == ord('q') or key == ord('Q') or key == 27:
                    print("Quitting...")
                    break
                
                frame_count += 1
                continue
            
            # Main tracking
            detected = self.detect_objects(frame)
            self.update_cart(detected, w)
            self.draw_ui(frame, detected)
            
            cv2.imshow(window_name, frame)
            
            # Handle keys
            key = cv2.waitKey(30)
            
            if key == ord('q') or key == ord('Q') or key == 27:
                print("\n👋 Quitting...")
                break
            
            elif key == ord('r') or key == ord('R'):
                self.stable_objects = {}
                self.cart_count = 0
                print("🔄 Cart reset!")
            
            elif key == ord('c') or key == ord('C'):
                self.is_calibrated = False
                self.stable_objects = {}
                self.cart_count = 0
                print("\n🔄 Recalibrating...")
            
            frame_count += 1
        
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"\n✅ Session complete!")
        print(f"Final cart: {self.cart_count} items")
        print(f"Total: ${self.cart_count * self.item_price:.2f}\n")

if __name__ == "__main__":
    try:
        tracker = SimpleShoppingCartTracker()
        tracker.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()