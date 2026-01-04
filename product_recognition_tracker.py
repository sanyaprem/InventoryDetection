import cv2
import numpy as np
from collections import defaultdict
import time

class CroppedProductTracker:
    def __init__(self):
        """Initialize tracker with ROI cropping"""
        print("Initializing Product Tracker with Cropping...")
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not self.cap.isOpened():
            raise Exception("Could not open webcam!")
        
        print("✅ Camera connected!")
        
        # Feature detector - LOWERED for poor camera quality
        self.detector = cv2.ORB_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Product database
        self.registered_products = {}
        self.next_product_id = 1
        self.max_products = 4
        
        # Detection state
        self.product_visible_frames = defaultdict(int)
        self.product_invisible_frames = defaultdict(int)
        self.product_is_visible = {}
        
        # Shopping cart
        self.picked_count = defaultdict(int)
        
        # Detection parameters - LOWERED for poor quality
        self.min_matches = 10  # Lowered from 15
        self.match_ratio = 0.8  # More lenient
        self.stable_visible_threshold = 15  # Lowered from 20
        self.stable_invisible_threshold = 15  # Lowered from 20
        
        # Mode
        self.mode = "REGISTRATION"
        self.registration_step = 0
        
        # ROI Selection for registration
        self.crop_box = None  # (x, y, w, h)
        self.crop_center = [640, 360]  # Center of crop box
        self.crop_size = [400, 500]  # Width, Height of crop box
        self.adjusting_size = False
        
        # Frame counter
        self.frame_count = 0
        
    def extract_features(self, image):
        """Extract features from image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # More aggressive denoising for poor camera
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Enhance contrast
        gray = cv2.equalizeHist(gray)
        
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def get_crop_box(self):
        """Get current crop box coordinates"""
        x = max(0, self.crop_center[0] - self.crop_size[0]//2)
        y = max(0, self.crop_center[1] - self.crop_size[1]//2)
        w = self.crop_size[0]
        h = self.crop_size[1]
        return (x, y, w, h)
    
    def adjust_crop_box(self, key):
        """Adjust crop box with keyboard"""
        move_step = 20
        size_step = 20
        
        if not self.adjusting_size:
            # Move crop box
            if key == ord('w') or key == ord('W'):
                self.crop_center[1] = max(self.crop_size[1]//2, self.crop_center[1] - move_step)
            elif key == ord('s') or key == ord('S'):
                self.crop_center[1] = min(720 - self.crop_size[1]//2, self.crop_center[1] + move_step)
            elif key == ord('a') or key == ord('A'):
                self.crop_center[0] = max(self.crop_size[0]//2, self.crop_center[0] - move_step)
            elif key == ord('d') or key == ord('D'):
                self.crop_center[0] = min(1280 - self.crop_size[0]//2, self.crop_center[0] + move_step)
        else:
            # Resize crop box
            if key == ord('w') or key == ord('W'):
                self.crop_size[1] = max(200, self.crop_size[1] - size_step)
            elif key == ord('s') or key == ord('S'):
                self.crop_size[1] = min(700, self.crop_size[1] + size_step)
            elif key == ord('a') or key == ord('A'):
                self.crop_size[0] = max(200, self.crop_size[0] - size_step)
            elif key == ord('d') or key == ord('D'):
                self.crop_size[0] = min(800, self.crop_size[0] + size_step)
        
        # Toggle size adjustment mode
        if key == ord('z') or key == ord('Z'):
            self.adjusting_size = not self.adjusting_size
    
    def register_product(self, frame, name, price, quantity):
        """Register product from cropped region only"""
        x, y, w, h = self.get_crop_box()
        
        # Crop to ROI only - NO BACKGROUND!
        cropped = frame[y:y+h, x:x+w].copy()
        
        if cropped.size == 0:
            print("❌ Crop region is empty!")
            return False
        
        # Extract features from CROPPED image only
        keypoints, descriptors = self.extract_features(cropped)
        
        if descriptors is None or len(keypoints) < 20:
            print(f"⚠️  Too few features: {len(keypoints) if keypoints else 0}. Need 20+!")
            print("   Try:")
            print("   - Adjust crop box to include ONLY the product")
            print("   - Better lighting")
            print("   - Move product closer")
            return False
        
        product_id = self.next_product_id
        self.registered_products[product_id] = {
            'id': product_id,
            'name': name,
            'price': price,
            'template': cropped,  # Store only cropped region!
            'keypoints': keypoints,
            'descriptors': descriptors,
            'initial_qty': quantity,
            'feature_count': len(keypoints)
        }
        
        # Initialize state
        self.product_is_visible[product_id] = False
        self.picked_count[name] = 0
        
        self.next_product_id += 1
        
        print(f"✅ Registered: {name} - ${price:.2f} - Qty: {quantity} - {len(keypoints)} features")
        return True
    
    def detect_products_in_frame(self, frame):
        """Detect products in frame"""
        if not self.registered_products:
            return {}
        
        keypoints_scene, descriptors_scene = self.extract_features(frame)
        
        if descriptors_scene is None:
            return {}
        
        detected = {}
        
        for product_id, product_data in self.registered_products.items():
            descriptors_product = product_data['descriptors']
            
            if descriptors_product is None:
                continue
            
            try:
                matches = self.matcher.knnMatch(descriptors_product, descriptors_scene, k=2)
            except:
                continue
            
            # Ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.match_ratio * n.distance:
                        good_matches.append(m)
            
            match_count = len(good_matches)
            
            if match_count >= self.min_matches:
                detected[product_id] = match_count
        
        return detected
    
    def update_tracking(self, detected_products):
        """Update product states"""
        for product_id in self.registered_products.keys():
            product_name = self.registered_products[product_id]['name']
            was_visible = self.product_is_visible.get(product_id, False)
            
            # Update counters
            if product_id in detected_products:
                self.product_visible_frames[product_id] += 1
                self.product_invisible_frames[product_id] = 0
            else:
                self.product_visible_frames[product_id] = 0
                self.product_invisible_frames[product_id] += 1
            
            # Determine state
            is_visible_now = False
            
            if self.product_visible_frames[product_id] >= self.stable_visible_threshold:
                is_visible_now = True
            elif self.product_invisible_frames[product_id] >= self.stable_invisible_threshold:
                is_visible_now = False
            else:
                is_visible_now = was_visible
            
            # Detect pick
            if was_visible and not is_visible_now:
                self.picked_count[product_name] += 1
                price = self.registered_products[product_id]['price']
                
                print(f"\n🛒 PICKED! {product_name}")
                print(f"   Price: ${price:.2f}")
                print(f"   Total {product_name}: {self.picked_count[product_name]}")
                print(f"   💰 Cart Total: ${self.get_total_cost():.2f}\n")
            
            self.product_is_visible[product_id] = is_visible_now
    
    def get_total_cost(self):
        """Calculate total"""
        total = 0.0
        for name, count in self.picked_count.items():
            for product in self.registered_products.values():
                if product['name'] == name:
                    total += count * product['price']
                    break
        return total
    
    def check_all_visible(self):
        """Check if all detected"""
        for product_id in self.registered_products.keys():
            if self.product_visible_frames[product_id] < self.stable_visible_threshold:
                return False
        return True
    
    def draw_crop_box(self, frame):
        """Draw adjustable crop box"""
        x, y, w, h = self.get_crop_box()
        
        # Draw crop box
        if self.adjusting_size:
            color = (0, 165, 255)  # Orange when resizing
            thickness = 4
        else:
            color = (0, 255, 0)  # Green when moving
            thickness = 3
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
        
        # Draw crosshair at center
        cx, cy = self.crop_center
        line_len = 20
        cv2.line(frame, (cx - line_len, cy), (cx + line_len, cy), color, 2)
        cv2.line(frame, (cx, cy - line_len), (cx, cy + line_len), color, 2)
        
        # Draw corners
        corner_len = 30
        # Top-left
        cv2.line(frame, (x, y), (x + corner_len, y), color, 3)
        cv2.line(frame, (x, y), (x, y + corner_len), color, 3)
        # Top-right
        cv2.line(frame, (x+w, y), (x+w - corner_len, y), color, 3)
        cv2.line(frame, (x+w, y), (x+w, y + corner_len), color, 3)
        # Bottom-left
        cv2.line(frame, (x, y+h), (x + corner_len, y+h), color, 3)
        cv2.line(frame, (x, y+h), (x, y+h - corner_len), color, 3)
        # Bottom-right
        cv2.line(frame, (x+w, y+h), (x+w - corner_len, y+h), color, 3)
        cv2.line(frame, (x+w, y+h), (x+w, y+h - corner_len), color, 3)
        
        # Show zoomed preview in corner
        if w > 0 and h > 0:
            cropped = frame[y:y+h, x:x+w].copy()
            if cropped.size > 0:
                # Resize preview to 300x300
                preview = cv2.resize(cropped, (300, 300))
                # Place in top-right corner
                frame[20:320, frame.shape[1]-320:frame.shape[1]-20] = preview
                cv2.rectangle(frame, (frame.shape[1]-320, 20), (frame.shape[1]-20, 320), (255, 255, 255), 2)
                cv2.putText(frame, "PREVIEW (what will be registered)", 
                           (frame.shape[1]-315, 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def draw_registration_ui(self, frame):
        """Draw registration UI with crop controls"""
        h, w = frame.shape[:2]
        
        # Draw crop box
        self.draw_crop_box(frame)
        
        # Instructions panel
        overlay = frame.copy()
        panel_h = 280
        cv2.rectangle(overlay, (10, h - panel_h - 10), (600, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        cv2.rectangle(frame, (10, h - panel_h - 10), (600, h - 10), (0, 255, 255), 3)
        
        y = h - panel_h + 20
        
        texts = [
            f"PRODUCT {self.registration_step + 1} OF {self.max_products}",
            "",
            "ADJUST CROP BOX:",
            "  W/A/S/D - Move box" if not self.adjusting_size else "  W/A/S/D - Resize box",
            "  Z - Toggle Move/Resize mode",
            "",
            "Position box around ONLY the product",
            "(exclude walls, floor, background)",
            "",
            "SPACE - Capture product",
            f"Progress: {self.registration_step}/{self.max_products}"
        ]
        
        for i, text in enumerate(texts):
            if i == 0:
                size, thickness, color = 0.75, 2, (0, 255, 255)
            elif i == 2:
                size, thickness, color = 0.65, 2, (255, 255, 0)
            elif "Toggle" in text:
                size, thickness, color = 0.55, 1, (0, 165, 255)
            else:
                size, thickness, color = 0.55, 1, (255, 255, 255)
            
            cv2.putText(frame, text, (25, y),
                       cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)
            y += 25
        
        # Mode indicator
        mode_text = "MODE: RESIZE (make box bigger/smaller)" if self.adjusting_size else "MODE: MOVE (position box)"
        mode_color = (0, 165, 255) if self.adjusting_size else (0, 255, 0)
        cv2.putText(frame, mode_text, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        # Registered products
        if self.registered_products:
            y_list = 80
            cv2.putText(frame, "Registered:", (20, y_list),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_list += 25
            for product in self.registered_products.values():
                text = f"  • {product['name']}: ${product['price']:.2f}"
                cv2.putText(frame, text, (20, y_list),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y_list += 23
    
    def draw_waiting_ui(self, frame):
        """Draw waiting UI"""
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (50, 50), (w - 50, 320), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        cv2.rectangle(frame, (50, 50), (w - 50, 320), (255, 165, 0), 4)
        
        y = 110
        cv2.putText(frame, "PLACE ALL PRODUCTS IN VIEW", (70, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 165, 0), 3)
        
        y += 60
        cv2.putText(frame, "Detecting products...", (70, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        y += 50
        
        for product_id, product_data in self.registered_products.items():
            name = product_data['name']
            frames = self.product_visible_frames[product_id]
            is_stable = frames >= self.stable_visible_threshold
            
            if is_stable:
                status = "✓ DETECTED"
                color = (0, 255, 0)
            else:
                status = f"Searching... {frames}/{self.stable_visible_threshold}"
                color = (255, 255, 0)
            
            text = f"  {name}: {status}"
            cv2.putText(frame, text, (90, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            y += 35
    
    def draw_tracking_ui(self, frame):
        """Draw tracking UI"""
        h, w = frame.shape[:2]
        
        # Product status
        y = 60
        cv2.putText(frame, "PRODUCT STATUS:", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        y += 35
        
        for product_id, product_data in self.registered_products.items():
            name = product_data['name']
            is_visible = self.product_is_visible.get(product_id, False)
            visible_frames = self.product_visible_frames[product_id]
            invisible_frames = self.product_invisible_frames[product_id]
            
            if is_visible:
                status = f"ON SHELF ({visible_frames}f)"
                color = (0, 255, 0)
            else:
                status = f"PICKED ({invisible_frames}f)"
                color = (0, 0, 255)
            
            text = f"  {name}: {status}"
            cv2.putText(frame, text, (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 28
        
        # Shopping cart
        cart_items = [item for item, count in self.picked_count.items() if count > 0]
        panel_h = 200 + (len(cart_items) * 38)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - panel_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        
        y = h - panel_h + 45
        
        cv2.putText(frame, "SHOPPING CART", (30, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
        
        y += 55
        
        if cart_items:
            cv2.putText(frame, "Items Picked:", (30, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
            y += 40
            
            for name in sorted(cart_items):
                count = self.picked_count[name]
                price = 0.0
                for prod in self.registered_products.values():
                    if prod['name'] == name:
                        price = prod['price']
                        break
                
                subtotal = count * price
                text = f"  {name}: {count} x ${price:.2f} = ${subtotal:.2f}"
                cv2.putText(frame, text, (45, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                y += 35
        else:
            cv2.putText(frame, "No items picked - all on shelf", (30, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            y += 40
        
        y += 15
        total = self.get_total_cost()
        cv2.rectangle(frame, (20, y - 8), (w - 20, y + 45), (255, 255, 255), 3)
        cv2.putText(frame, f"TOTAL: ${total:.2f}", (35, y + 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
    
    def get_product_details(self, num):
        """Get details from terminal"""
        print(f"\n{'='*70}")
        print(f"PRODUCT {num + 1} REGISTRATION")
        print(f"{'='*70}")
        
        name = input("Product name: ").strip()
        while not name:
            name = input("Product name: ").strip()
        
        while True:
            try:
                price = float(input("Price: $"))
                if price > 0:
                    break
            except ValueError:
                pass
        
        qty_input = input("Quantity (default 1): ").strip()
        quantity = int(qty_input) if qty_input else 1
        
        return name, price, quantity
    
    def run(self):
        """Main loop"""
        print("\n" + "="*70)
        print("PRODUCT TRACKER - WITH CROP/ZOOM")
        print("="*70)
        print("\n🎯 NEW FEATURES:")
        print("✓ ADJUSTABLE CROP BOX - frame only the product!")
        print("✓ Excludes walls/floor/background")
        print("✓ Live preview shows what will be registered")
        print("✓ Optimized for poor camera quality")
        print("\n📝 REGISTRATION:")
        print("1. Adjust GREEN box around product (W/A/S/D)")
        print("2. Press Z to switch between Move/Resize")
        print("3. Check preview (top-right)")
        print("4. Press SPACE to capture")
        print("\n🔧 LOWERED THRESHOLDS:")
        print("- Better for low quality cameras")
        print("- More lenient matching")
        print("="*70 + "\n")
        
        window_name = 'Product Tracker - Cropped'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        time.sleep(2)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # REGISTRATION
            if self.mode == "REGISTRATION":
                self.draw_registration_ui(frame)
                cv2.imshow(window_name, frame)
                
                key = cv2.waitKey(30)
                
                # Adjust crop box
                if key in [ord('w'), ord('W'), ord('a'), ord('A'), 
                          ord('s'), ord('S'), ord('d'), ord('D'), 
                          ord('z'), ord('Z')]:
                    self.adjust_crop_box(key)
                
                elif key == 32:  # SPACE
                    name, price, qty = self.get_product_details(self.registration_step)
                    
                    print("\n📸 Capturing cropped region...")
                    ret, cap = self.cap.read()
                    if ret:
                        cap = cv2.flip(cap, 1)
                        if self.register_product(cap, name, price, qty):
                            self.registration_step += 1
                            
                            if self.registration_step >= self.max_products:
                                self.mode = "WAITING"
                                print("\n" + "="*70)
                                print("✅ ALL REGISTERED!")
                                print("="*70)
                                print("\nPlace all products in view...")
                                print("="*70 + "\n")
                        else:
                            print("❌ Failed. Adjust crop box and try again.")
                
                elif key == ord('q') or key == 27:
                    break
            
            # WAITING
            elif self.mode == "WAITING":
                detected = self.detect_products_in_frame(frame)
                
                for product_id in self.registered_products.keys():
                    if product_id in detected:
                        self.product_visible_frames[product_id] += 1
                    else:
                        self.product_visible_frames[product_id] = 0
                
                if self.check_all_visible():
                    self.mode = "TRACKING"
                    for product_id in self.registered_products.keys():
                        self.product_is_visible[product_id] = True
                    
                    print("\n" + "="*70)
                    print("✅ TRACKING STARTED!")
                    print("="*70 + "\n")
                
                self.draw_waiting_ui(frame)
                cv2.imshow(window_name, frame)
                
                key = cv2.waitKey(30)
                if key == ord('q') or key == 27:
                    break
            
            # TRACKING
            else:
                detected = self.detect_products_in_frame(frame)
                self.update_tracking(detected)
                
                self.draw_tracking_ui(frame)
                
                visible = sum(1 for v in self.product_is_visible.values() if v)
                status = f"Visible: {visible}/4 | Cart: ${self.get_total_cost():.2f}"
                cv2.putText(frame, status, (10, h - 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow(window_name, frame)
                
                key = cv2.waitKey(30)
                
                if key == ord('q') or key == 27:
                    break
                elif key == ord('r') or key == ord('R'):
                    self.picked_count = defaultdict(int)
                    print("\n🔄 Cart reset!\n")
            
            self.frame_count += 1
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Receipt
        print("\n" + "="*70)
        print("FINAL RECEIPT")
        print("="*70)
        
        if any(count > 0 for count in self.picked_count.values()):
            print("\nItems:")
            for name, count in sorted(self.picked_count.items()):
                if count > 0:
                    price = 0.0
                    for prod in self.registered_products.values():
                        if prod['name'] == name:
                            price = prod['price']
                            break
                    print(f"  {name}: {count} x ${price:.2f} = ${count * price:.2f}")
            
            print(f"\n{'─'*50}")
            print(f"TOTAL: ${self.get_total_cost():.2f}")
            print(f"{'─'*50}\n")
        else:
            print("\nNo items\n")
        
        print("="*70 + "\n")

if __name__ == "__main__":
    try:
        tracker = CroppedProductTracker()
        tracker.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()