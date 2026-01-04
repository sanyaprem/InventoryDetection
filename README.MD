# Product Tracker with Crop/Zoom - Guide

## 🎯 What's New

### Problem Solved:
- ❌ Background (walls, floor) being registered
- ❌ Products far from camera
- ❌ Poor camera quality causing false matches

### Solution:
- ✅ **Interactive crop box** - frame ONLY the product
- ✅ **Live preview** - see exactly what will be registered
- ✅ **Keyboard controls** - precise positioning
- ✅ **Lowered thresholds** - works with poor cameras

---

## 🎮 How to Use Crop Box

### During Registration:

**You'll see a GREEN BOX on screen**

This box controls what gets registered - ONLY what's inside the box!

### Controls:

```
┌─────────────────────────────────┐
│  MOVEMENT MODE (Default)        │
├─────────────────────────────────┤
│  W - Move box UP                │
│  S - Move box DOWN              │
│  A - Move box LEFT              │
│  D - Move box RIGHT             │
└─────────────────────────────────┘

Press Z to switch to:

┌─────────────────────────────────┐
│  RESIZE MODE (Orange box)       │
├─────────────────────────────────┤
│  W - Make box SHORTER           │
│  S - Make box TALLER            │
│  A - Make box NARROWER          │
│  D - Make box WIDER             │
└─────────────────────────────────┘

Press Z again to go back to MOVEMENT MODE
```

---

## 📋 Step-by-Step Registration

### 1. Position Your Product

```
Place product in front of camera
(can be far away - crop will zoom in!)
```

### 2. Adjust Crop Box

**GREEN BOX = Movement Mode**
```
Use W/A/S/D to move the box
Position it over your product
```

**ORANGE BOX = Resize Mode (press Z)**
```
Use W/A/S/D to resize the box
Make it fit JUST around the product
```

### 3. Check Preview (Top-Right Corner)

```
┌──────────────────┐
│ PREVIEW          │
│                  │
│ (zoomed view)    │
│                  │
│ Shows exactly    │
│ what will be     │
│ registered       │
└──────────────────┘
```

**Good preview:**
- ✅ Product fills most of the box
- ✅ No wall/floor visible
- ✅ Product is clear

**Bad preview:**
- ❌ Too much background
- ❌ Product too small
- ❌ Product cut off

### 4. Capture

```
When crop box looks good:
Press SPACE
```

---

## 🎨 Visual Guide

### Example: Registering Pringles Can

**Step 1: Initial view**
```
┌─────────────────────────────────┐
│                                 │
│                                 │
│      [Pringles can]             │
│         (far away)              │
│                                 │
│                                 │
│   ┌──GREEN BOX──┐               │
│   │             │               │
│   │             │               │
│   └─────────────┘               │
└─────────────────────────────────┘
Box is in center but not on product!
```

**Step 2: Move box (W/A/S/D)**
```
┌─────────────────────────────────┐
│                                 │
│      ┌──GREEN BOX──┐            │
│      │ [Pringles]  │            │
│      │   can       │            │
│      └─────────────┘            │
│                                 │
│                                 │
└─────────────────────────────────┘
Box positioned over product!
```

**Step 3: Resize (Z then W/A/S/D)**
```
┌─────────────────────────────────┐
│                                 │
│      ┌ORANGE BOX─┐              │
│      │ Pringles  │              │
│      │   can     │              │
│      │  (fits!)  │              │
│      └───────────┘              │
│                                 │
└─────────────────────────────────┘
Box fits product perfectly!
```

**Step 4: Check preview**
```
Top-right shows:
┌────────────┐
│  PRINGLES  │  ← Clear, zoomed
│    CAN     │  ← No background!
│            │  ← Perfect! ✓
└────────────┘
```

**Step 5: Press SPACE to capture!**

---

## 🔧 Settings for Poor Camera Quality

**Already adjusted in code:**

```python
# More features extracted
nfeatures=2000  (was 1500)

# Lower matching threshold
min_matches=10  (was 15)

# More lenient ratio
match_ratio=0.8  (was 0.75)

# Faster confirmation
stable_threshold=15  (was 20)

# Better noise reduction
GaussianBlur=(7,7)  (was 5,5)

# Contrast enhancement
equalizeHist=True  (new!)
```

**These help with:**
- ✅ Blurry cameras
- ✅ Poor lighting
- ✅ Low resolution
- ✅ Grainy images

---

## 💡 Tips for Best Results

### Product Distance

**Too far:**
```
Camera sees: [tiny product] + lots of background
Problem: Too much background noise
```

**Too close:**
```
Camera sees: [product fills screen] but blurry
Problem: Can't see full product
```

**Just right:**
```
Camera sees: [clear product] taking up 50-70% of frame
Solution: Use crop box to zoom in!
```

### Crop Box Strategy

1. **First registration:**
   - Start with default box
   - Move to product
   - Resize to fit SNUGLY
   - Leave small margin (10-20 pixels)

2. **Subsequent registrations:**
   - Box stays at last position
   - Just move to new product
   - Adjust size if needed

### Background Elimination

**Include:**
- ✅ Entire product
- ✅ Product label/text
- ✅ Product edges

**Exclude:**
- ❌ Walls
- ❌ Floor
- ❌ Other objects
- ❌ Your hands (during capture)

---

## 📊 On-Screen Indicators

### Crop Box Colors

| Color | Mode | Action |
|-------|------|--------|
| **GREEN** | Movement | Move box with W/A/S/D |
| **ORANGE** | Resize | Change size with W/A/S/D |

### Corner Markers

```
┌─────────────┐
│             │  ← Corners show crop area
│   PRODUCT   │
│             │
└─────────────┘
```

### Crosshair

```
        │
        │
────────+────────  ← Center point
        │
        │
```

Shows exact center of crop box

---

## 🐛 Troubleshooting

### "Too few features" Error

**Problem:** Less than 20 features detected

**Solutions:**

1. **Adjust crop box:**
   - Include more of the product
   - Make box bigger
   - Ensure product label is visible

2. **Better lighting:**
   - Turn on more lights
   - Avoid shadows
   - Position near window

3. **Move closer:**
   - Bring product closer to camera
   - Or make crop box bigger

4. **Better product:**
   - Use products with text/logos
   - Avoid plain items

### Products Not Detected After Registration

**Problem:** 0/4 detected in waiting mode

**Solutions:**

1. **Same distance:**
   - Keep products at same distance as registration
   - Don't move camera

2. **Same orientation:**
   - Product facing same way
   - Not rotated or upside down

3. **Same lighting:**
   - Lighting conditions match registration
   - No new shadows

4. **Check if registered correctly:**
   - Did you get "✅ Registered" message?
   - Did it show enough features?

### Still Glitching/Flickering

**If detection is unstable:**

1. **Increase stability threshold:**
   ```python
   # Line 40-41
   self.stable_visible_threshold = 20  # Up from 15
   self.stable_invisible_threshold = 20
   ```

2. **Increase minimum matches:**
   ```python
   # Line 38
   self.min_matches = 15  # Up from 10
   ```

3. **Better feature matching:**
   ```python
   # Line 39
   self.match_ratio = 0.75  # Down from 0.8 (stricter)
   ```

---

## 🎯 Example Session

```bash
$ python cropped_product_tracker.py

# PRODUCT 1 - Pringles

[GREEN box appears in center]

Press D D D (move right to Pringles)
Press S S (move down a bit)
Press Z (switch to resize mode)
[Box turns ORANGE]
Press S S (make taller)
Press A (make narrower)
Press Z (back to move mode)
[Box turns GREEN]

Check preview (top-right): ✓ Looks good!

Press SPACE

Terminal asks:
  Product name: Pringles
  Price: $4.50
  Quantity: 1

✅ Registered: Pringles - $4.50 - Qty: 1 - 234 features

# PRODUCT 2 - Coca Cola

[Move box to Coke, adjust, SPACE, enter details]
✅ Registered: Coca Cola - $2.50 - Qty: 1 - 189 features

# PRODUCT 3 - Doritos

[Move box to Doritos, adjust, SPACE, enter details]
✅ Registered: Doritos - $3.99 - Qty: 1 - 276 features

# PRODUCT 4 - Sprite

[Move box to Sprite, adjust, SPACE, enter details]
✅ Registered: Sprite - $2.20 - Qty: 1 - 198 features

✅ ALL REGISTERED!

# WAITING MODE

Place all products in view...

  Pringles: Searching... 8/15
  Coca Cola: Searching... 5/15
  Doritos: ✓ DETECTED
  Sprite: Searching... 12/15

[Wait a moment...]

  Pringles: ✓ DETECTED
  Coca Cola: ✓ DETECTED
  Doritos: ✓ DETECTED
  Sprite: ✓ DETECTED

✅ TRACKING STARTED!

# TRACKING

[Pick Pringles]
🛒 PICKED! Pringles
   Price: $4.50
   💰 Cart Total: $4.50

[Pick Coke]
🛒 PICKED! Coca Cola
   Price: $2.50
   💰 Cart Total: $7.00
```

---

## ⌨️ All Controls

### Registration Mode

| Key | Action |
|-----|--------|
| **W/A/S/D** | Move or resize crop box |
| **Z** | Toggle Move/Resize mode |
| **SPACE** | Capture product |
| **Q** | Quit |

### Waiting/Tracking Mode

| Key | Action |
|-----|--------|
| **Q** | Quit |
| **R** | Reset cart (tracking only) |

---

## ✅ Why This Fixes Your Issues

### Issue 1: Background Noise
**Before:** Registered whole frame (walls, floor, etc.)  
**Now:** Only register cropped product area ✓

### Issue 2: Products Far Away
**Before:** Small product + lots of background  
**Now:** Crop box zooms in on product ✓

### Issue 3: Poor Camera Quality
**Before:** High thresholds, strict matching  
**Now:** Lowered thresholds, lenient matching ✓

### Issue 4: Glitching Detection
**Before:** Background features causing false matches  
**Now:** Clean product-only registration ✓

---

## 🎉 This Should Work Much Better!

**Key improvements:**
1. ✅ Crop box eliminates background
2. ✅ Preview shows exact registration
3. ✅ Optimized for poor cameras
4. ✅ Easy keyboard controls

Try it and let me know if the cropping helps!