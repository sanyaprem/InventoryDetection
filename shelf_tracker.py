import cv2
import numpy as np
from collections import defaultdict
import time

class SimpleShelfTracker:
    def __init__(self):
        """Initialize simple shelf inventory tracker"""
        print("Initializing Simple Shelf Tracker...")
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not self.cap.isOpened():
            raise Exception("Could not open webcam!")
        
        print("✅ Camera connected!")
        
        # Shelf region (top 40% of frame)
        self.shelf_region = None
        self.shelf_height_ratio = 0.4
        
        # Product tracking
        self.shelf_products = {}  # Currently on shelf
        self.picked_products = defaultdict(int)  # Count of picked items by color
        
        # Product colors and pricing
        self.product_colors = {
            'red': ([0, 100, 100], [10, 255, 255]),
            'blue': ([100, 100, 100], [130, 255, 255]),
            'green': ([40, 100, 100], [80, 255, 255]),
            'yellow': ([20, 100, 100], [35, 255, 255]),
        }
        
        self.product_prices = {
            'red': 15.99,
            'blue': 12.99,
            'green': 9.99,
            'yellow': 18.99,
        }
        
        # Detection parameters
        self.min_product_area = 1500
        self.max_product_area = 25000
        
        # Product tracking (to detect picks)
        self.product_id_counter = 1
        self.product_history = {}  # Track products over time
        self.frames_missing_threshold = 5  # Frames before confirming pick
        
        # Total billing
        self.total_cost = 0.0
        
    def setup_shelf_region(self, frame_height, frame_width):
        """Define shelf region"""
        shelf_height = int(frame_height * self.shelf_height_ratio)
        
        self.shelf_region = (
            0,              # x1
            0,              # y1
            frame_width,    # x2
            shelf_height    # y2
        )
        
        print(f"✅ Shelf region: Top {self.shelf_height_ratio*100}% of frame")
    
    def detect_products_on_shelf(self, frame):
        """Detect products on shelf using color detection"""
        if self.shelf_region is None:
            return []
        
        # Extract shelf region
        x1, y1, x2, y2 = self.shelf_region
        shelf_roi = frame[y1:y2, x1:x2]
        
        hsv = cv2.cvtColor(shelf_roi, cv2.COLOR_BGR2HSV)
        detected_products = []
        
        for color_name, (lower, upper) in self.product_colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Clean up mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if self.min_product_area < area < self.max_product_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Convert back to full frame coordinates
                    x += x1
                    y += y1
                    
                    center = (x + w//2, y + h//2)
                    
                    detected_products.append({
                        'bbox': (x, y, w, h),
                        'center': center,
                        'color': color_name,
                        'area': area
                    })
        
        return detected_products
    
    def distance(self, p1, p2):
        """Calculate distance between two points"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def track_shelf_inventory(self, detected_products):
        """Track products and detect when they're picked"""
        current_frame_products = {}
        detected_ids = set()
        
        # Match detected products to tracked products
        for product in detected_products:
            center = product['center']
            bbox = product['bbox']
            color = product['color']
            
            # Find closest tracked product of same color
            min_dist = float('inf')
            matched_id = None
            
            for product_id, tracked in self.product_history.items():
                if tracked['color'] == color and tracked.get('on_shelf', True):
                    dist = self.distance(center, tracked['center'])
                    if dist < min_dist and dist < 100:  # Products don't move much
                        min_dist = dist
                        matched_id = product_id
            
            if matched_id is not None:
                # Update existing product
                current_frame_products[matched_id] = {
                    'bbox': bbox,
                    'center': center,
                    'color': color,
                    'on_shelf': True,
                    'frames_missing': 0
                }
                detected_ids.add(matched_id)
            else:
                # New product on shelf
                new_id = self.product_id_counter
                current_frame_products[new_id] = {
                    'bbox': bbox,
                    'center': center,
                    'color': color,
                    'on_shelf': True,
                    'frames_missing': 0
                }
                detected_ids.add(new_id)
                self.product_id_counter += 1
        
        # Check for products that disappeared (picked)
        for product_id, product_data in self.product_history.items():
            if product_id not in detected_ids and product_data.get('on_shelf', True):
                # Product not detected this frame
                frames_missing = product_data.get('frames_missing', 0) + 1
                
                if frames_missing >= self.frames_missing_threshold:
                    # Product was picked!
                    self.on_product_picked(product_data)
                    # Mark as picked (don't track anymore)
                    current_frame_products[product_id] = {
                        **product_data,
                        'on_shelf': False,
                        'frames_missing': frames_missing
                    }
                else:
                    # Still might be there, just briefly occluded
                    current_frame_products[product_id] = {
                        **product_data,
                        'frames_missing': frames_missing
                    }
        
        self.product_history = current_frame_products
        
        # Update shelf_products (only items currently visible)
        self.shelf_products = {
            pid: data for pid, data in current_frame_products.items()
            if data.get('on_shelf', True) and data.get('frames_missing', 0) == 0
        }
    
    def on_product_picked(self, product_data):
        """Handle when a product is picked"""
        color = product_data['color']
        price = self.product_prices.get(color, 10.00)
        
        # Increment count
        self.picked_products[color] += 1
        
        # Add to total
        self.total_cost += price
        
        print(f"\n🛒 PICKED! {color.upper()} item")
        print(f"   Price: ${price:.2f}")
        print(f"   Total {color} items: {self.picked_products[color]}")
        print(f"   💰 Running Total: ${self.total_cost:.2f}\n")
    
    def draw_shelf_region(self, frame):
        """Draw shelf region boundary"""
        if self.shelf_region:
            x1, y1, x2, y2 = self.shelf_region
            
            # Draw shelf box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            
            # Label
            cv2.putText(frame, "SHELF AREA - Place Items Here", (x1 + 20, y1 + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    
    def draw_shelf_products(self, frame):
        """Draw products currently on shelf"""
        for product_id, product_data in self.shelf_products.items():
            x, y, w, h = product_data['bbox']
            color_name = product_data['color']
            price = self.product_prices.get(color_name, 10.00)
            
            # Draw product box
            box_color = (255, 255, 0)  # Cyan
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            
            # Label with color and price
            label = f"{color_name.upper()} ${price:.2f}"
            cv2.putText(frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            
            # Draw center point
            center = product_data['center']
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
    
    def draw_shopping_summary(self, frame):
        """Draw shopping cart summary panel"""
        h, w = frame.shape[:2]
        
        # Calculate panel height based on items picked
        num_product_types = len(self.picked_products)
        panel_h = 200 + (num_product_types * 40)
        
        # Semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - panel_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        y = h - panel_h + 40
        
        # Title
        cv2.putText(frame, "SHOPPING CART", (30, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        y += 50
        
        # Items picked
        if self.picked_products:
            cv2.putText(frame, "Items Picked:", (30, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            y += 35
            
            for color, count in sorted(self.picked_products.items()):
                price = self.product_prices[color]
                subtotal = count * price
                
                item_text = f"  {color.upper()}: {count} x ${price:.2f} = ${subtotal:.2f}"
                cv2.putText(frame, item_text, (40, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y += 35
        else:
            cv2.putText(frame, "No items picked yet", (30, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            y += 40
        
        # Total
        y += 10
        cv2.rectangle(frame, (20, y - 5), (w - 20, y + 45), (255, 255, 255), 2)
        
        total_text = f"TOTAL: ${self.total_cost:.2f}"
        cv2.putText(frame, total_text, (30, y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    
    def draw_stats(self, frame):
        """Draw real-time stats"""
        h, w = frame.shape[:2]
        
        # Stats
        products_on_shelf = len(self.shelf_products)
        total_picked = sum(self.picked_products.values())
        
        stats = [
            f"On Shelf: {products_on_shelf}",
            f"Picked: {total_picked}",
            f"Total: ${self.total_cost:.2f}"
        ]
        
        y = 90
        for stat in stats:
            cv2.putText(frame, stat, (w - 250, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += 30
    
    def draw_instructions(self, frame):
        """Draw usage instructions"""
        h, w = frame.shape[:2]
        
        instructions = [
            "INSTRUCTIONS:",
            "1. Place colored items in SHELF area (top)",
            "2. Pick items from shelf",
            "3. System detects and counts automatically!",
            "",
            "PRICING:",
            "RED: $15.99  |  BLUE: $12.99",
            "GREEN: $9.99  |  YELLOW: $18.99",
            "",
            "Q - Quit  |  R - Reset Cart"
        ]
        
        y = h - len(instructions) * 22 - 250
        for i, text in enumerate(instructions):
            color = (0, 255, 255) if i == 0 or i == 5 else (220, 220, 220)
            thickness = 2 if i == 0 or i == 5 else 1
            size = 0.5 if i == 0 or i == 5 else 0.45
            
            cv2.putText(frame, text, (w - 450, y),
                       cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)
            y += 22
    
    def run(self):
        """Main loop"""
        print("\n" + "="*70)
        print("SIMPLE SHELF INVENTORY TRACKER")
        print("="*70)
        print("\n🎯 CONCEPT:")
        print("Track what's picked from shelf - simple and effective!")
        print("\n✨ FEATURES:")
        print("✓ No calibration needed")
        print("✓ Detects products on shelf")
        print("✓ Counts picks automatically")
        print("✓ Tracks quantity per product")
        print("✓ Calculates total cost")
        print("✓ Simple single-person use")
        print("\n📝 SETUP:")
        print("1. Place COLORED items in TOP area (shelf)")
        print("2. Items will be detected automatically")
        print("3. Pick items - system counts them!")
        print("\n💰 PRICING:")
        print("  RED items:    $15.99")
        print("  BLUE items:   $12.99")
        print("  GREEN items:  $9.99")
        print("  YELLOW items: $18.99")
        print("\n⌨️  CONTROLS:")
        print("  Q or ESC - Quit")
        print("  R - Reset shopping cart")
        print("="*70 + "\n")
        
        window_name = 'Simple Shelf Tracker'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        print("⏳ Starting in 2 seconds...")
        time.sleep(2)
        print("✅ System ready! Place items on shelf and start picking!\n")
        
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Mirror frame
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Setup shelf region on first frame
            if self.shelf_region is None:
                self.setup_shelf_region(h, w)
            
            # Detect products on shelf
            detected_products = self.detect_products_on_shelf(frame)
            
            # Track inventory
            self.track_shelf_inventory(detected_products)
            
            # Draw everything
            self.draw_shelf_region(frame)
            self.draw_shelf_products(frame)
            self.draw_shopping_summary(frame)
            self.draw_stats(frame)
            self.draw_instructions(frame)
            
            # Status
            status = f"Frame: {frame_count} | Products: {len(self.shelf_products)} | Total Picked: {sum(self.picked_products.values())}"
            cv2.putText(frame, status, (10, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow(window_name, frame)
            
            # Keyboard
            key = cv2.waitKey(30)
            
            if key == ord('q') or key == ord('Q') or key == 27:
                print("\n👋 Quitting...")
                break
            
            elif key == ord('r') or key == ord('R'):
                # Reset cart but keep tracking shelf items
                self.picked_products = defaultdict(int)
                self.total_cost = 0.0
                print("\n🔄 Shopping cart reset!\n")
            
            frame_count += 1
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Final receipt
        print("\n" + "="*70)
        print("FINAL RECEIPT")
        print("="*70)
        
        if self.picked_products:
            print("\nItems Purchased:")
            for color, count in sorted(self.picked_products.items()):
                price = self.product_prices[color]
                subtotal = count * price
                print(f"  {color.upper()}: {count} x ${price:.2f} = ${subtotal:.2f}")
            
            print(f"\n{'─'*40}")
            print(f"TOTAL: ${self.total_cost:.2f}")
            print(f"{'─'*40}\n")
        else:
            print("\nNo items purchased\n")
        
        print("="*70 + "\n")

if __name__ == "__main__":
    try:
        tracker = SimpleShelfTracker()
        tracker.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()