# Save as test_keys.py
import cv2
import numpy as np

print("Testing keyboard input...")
cv2.namedWindow('Test', cv2.WINDOW_NORMAL)

while True:
    img = np.zeros((300, 600, 3), dtype=np.uint8)
    cv2.putText(img, "CLICK HERE, then press Q", (100, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow('Test', img)
    
    key = cv2.waitKey(30)
    if key != -1:
        print(f"✅ Key detected! Code: {key}")
        if key == ord('q') or key == 27:
            print("Quitting!")
            break

cv2.destroyAllWindows()
print("Test complete!")